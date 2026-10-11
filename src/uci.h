// uci.h : UCI protocol front end for GrandMassive.
//
// Handles the GUI side of the protocol: reading commands, storing the
// position (start FEN + move list), parsing "go" limits, working out a time
// budget, and running the search on a worker thread so "stop", "isready" and
// "ponderhit" are answered while it thinks.
//
// Board representation and move generation come from Chess.h
// (Disservin chess-library); the search itself lives in Search.h.
// run_search() below connects the two: it hands the search the board and
// UCI limits, and turns its results into "info" / "bestmove" lines.
//
// Supported GUI -> engine commands:
//   uci, isready, ucinewgame, setoption, position, go, stop, ponderhit, quit
//   debug / register (accepted, ignored)
//   d    (non-standard: print position state)
//   eval (non-standard: print the evaluation breakdown)
//   book (non-standard: list book moves for the current position)
//   bench [depth] (non-standard: fixed-depth search of BenchFens, prints
//                  total nodes and nps; also runs as "GrandMassive bench")
//   orderstats [depth] (non-standard: bench at depth 13 plus move-ordering
//                  and pruning statistics; "GrandMassive orderstats")
//   testsuite <file.epd> [ms] (non-standard: solve an EPD test suite such as
//                  WAC, ms per position, default 1000)
//
// go parameters: depth, wtime, btime, winc, binc, movestogo, movetime,
//                nodes, mate, infinite, ponder, searchmoves, perft

#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <optional>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "Chess.h"
#include "Book.h"
#include "Search.h"

namespace uci {

// ---------------------------------------------------------------------------
// Engine identity / constants
// ---------------------------------------------------------------------------
inline constexpr std::string_view EngineName   = "GrandMassive";
inline constexpr std::string_view EngineAuthor = "Warmedpie";
inline constexpr std::string_view StartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

inline constexpr int MaxDepth = 128;

using TimeMs = std::int64_t;

inline TimeMs now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Output: every line goes through here so the search thread and the input
// thread never interleave, and every line is flushed (GUIs read line by line).
// ---------------------------------------------------------------------------
inline std::mutex& io_mutex() {
    static std::mutex m;
    return m;
}

template <class... Args>
void send(Args&&... args) {
    std::lock_guard lock(io_mutex());
    (std::cout << ... << std::forward<Args>(args)) << std::endl;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Long algebraic move as UCI uses it: e2e4, e7e8q, or the null move 0000.
inline bool is_move_token(std::string_view s) {
    if (s == "0000") return true;
    if (s.size() != 4 && s.size() != 5) return false;
    auto file = [](char c) { return c >= 'a' && c <= 'h'; };
    auto rank = [](char c) { return c >= '1' && c <= '8'; };
    if (!file(s[0]) || !rank(s[1]) || !file(s[2]) || !rank(s[3])) return false;
    return s.size() == 4 || std::string_view("qrbn").find(s[4]) != std::string_view::npos;
}

template <class T>
bool parse_int(std::string_view s, T& out) {
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc() && ptr == s.data() + s.size();
}

// Find the legal move whose UCI string matches `text`; NO_MOVE if none does.
inline chess::Move find_legal_move(const chess::Board& board, std::string_view text) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& m : moves)
        if (chess::uci::moveToUci(m, board.chess960()) == text) return m;
    return chess::Move(chess::Move::NO_MOVE);
}

// Minimal FEN sanity check: exactly one king per side and the side not to
// move is not in check. Guards against garbage FENs from the GUI.
inline bool board_is_sane(const chess::Board& board) {
    using chess::Color;
    using chess::PieceType;
    if (board.pieces(PieceType::KING, Color::WHITE).count() != 1) return false;
    if (board.pieces(PieceType::KING, Color::BLACK).count() != 1) return false;
    const Color them = ~board.sideToMove();
    return !board.isAttacked(board.kingSq(them), board.sideToMove());
}

// ---------------------------------------------------------------------------
// Bench: a fixed set of positions (mostly Stockfish's benchmark set) searched
// to a fixed depth. The total node count is a fingerprint of the search: a
// change meant to be speed-only must leave it unchanged, and any functional
// change will move it. nps compares the speed of two builds.
// ---------------------------------------------------------------------------
inline constexpr int BenchDepth = 7;

inline constexpr std::string_view BenchFens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
};

// ---------------------------------------------------------------------------
// EPD test suites (WAC, ECM, STS, ...): one position per line, a 4-field FEN
// followed by operations such as  bm Qg6; am Rd6; id "WAC.001"; ce 150;
//   bm = best move(s), am = move(s) to avoid, ce = reference eval (cp, side
//   to move). Moves are in SAN.
// ---------------------------------------------------------------------------
struct EpdEntry {
    std::string fen;
    std::string id;
    std::vector<std::string> bm, am;
    std::optional<int> ce;
};

// SAN without check/mate/annotation marks, castling spelled with letters O.
inline std::string normalize_san(std::string san) {
    while (!san.empty() && std::string_view("+#!?").find(san.back()) != std::string_view::npos)
        san.pop_back();
    std::replace(san.begin(), san.end(), '0', 'O');
    return san;
}

// Legal move matching a SAN string, or NO_MOVE. Parses the SAN itself
// (piece, optional from-file / from-rank hint, target, promotion) rather
// than comparing against chess::uci::moveToSan, which doesn't disambiguate
// (it writes both rooks' moves as "Rxg6+" where the EPD says "Rfxg6+").
inline chess::Move find_san_move(const chess::Board& board, const std::string& raw) {
    const chess::Move none(chess::Move::NO_MOVE);
    std::string san = normalize_san(raw);
    san.erase(std::remove(san.begin(), san.end(), 'x'), san.end());
    san.erase(std::remove(san.begin(), san.end(), '='), san.end());

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    // Castling: compare against the library's own castling notation.
    if (san == "OO" || san == "O-O" || san == "OOO" || san == "O-O-O") {
        const bool queenside = san == "OOO" || san == "O-O-O";
        for (const auto& m : moves)
            if (m.typeOf() == chess::Move::CASTLING && ((m.to() < m.from()) == queenside)) return m;
        return none;
    }

    // Piece letter (pawn if none).
    chess::PieceType piece = chess::PieceType::PAWN;
    std::size_t i = 0;
    if (!san.empty() && std::string_view("KQRBN").find(san[0]) != std::string_view::npos) {
        piece = chess::PieceType(std::string_view(&san[0], 1));
        i = 1;
    }

    // Promotion piece (pawns only): trailing Q/R/B/N.
    chess::PieceType promo = chess::PieceType::NONE;
    if (piece == chess::PieceType::PAWN && !san.empty() &&
        std::string_view("QRBN").find(san.back()) != std::string_view::npos) {
        promo = chess::PieceType(std::string_view(&san.back(), 1));
        san.pop_back();
    }

    if (san.size() < i + 2) return none;
    const std::string target = san.substr(san.size() - 2);
    const std::string hint = san.substr(i, san.size() - 2 - i);  // "", "f", "1", or "f1"
    const chess::Square to(target);
    if (!to.is_valid()) return none;

    chess::Move found = none;
    for (const auto& m : moves) {
        if (m.typeOf() == chess::Move::CASTLING) continue;
        if (board.at(m.from()).type() != piece || m.to() != to) continue;
        if ((m.typeOf() == chess::Move::PROMOTION ? m.promotionType() : chess::PieceType::NONE) != promo)
            continue;
        const std::string from = static_cast<std::string>(m.from());
        bool match = true;
        for (char c : hint) {
            if (c >= 'a' && c <= 'h') match &= (from[0] == c);
            else if (c >= '1' && c <= '8') match &= (from[1] == c);
            else match = false;
        }
        if (!match) continue;
        if (found != none) return none;  // ambiguous
        found = m;
    }
    return found;
}

inline bool parse_epd(std::string_view line, EpdEntry& e) {
    std::istringstream is{std::string(line)};
    std::string f[4];
    if (!(is >> f[0] >> f[1] >> f[2] >> f[3])) return false;
    e = {};
    e.fen = f[0] + ' ' + f[1] + ' ' + f[2] + ' ' + f[3] + " 0 1";

    // Operations are ';'-separated; quoted operands may contain spaces.
    std::string rest;
    std::getline(is, rest);
    std::vector<std::string> ops;
    std::string cur;
    bool quoted = false;
    for (char c : rest) {
        if (c == '"') quoted = !quoted;
        if (c == ';' && !quoted) { ops.push_back(cur); cur.clear(); }
        else cur += c;
    }
    ops.push_back(cur);

    for (const auto& op : ops) {
        std::istringstream os(op);
        std::string code, arg;
        if (!(os >> code)) continue;
        std::vector<std::string> args;
        while (os >> arg) args.push_back(arg);
        if (code == "bm") e.bm = args;
        else if (code == "am") e.am = args;
        else if (code == "id") {
            std::string id;
            for (const auto& a : args) id += (id.empty() ? "" : " ") + a;
            id.erase(std::remove(id.begin(), id.end(), '"'), id.end());
            e.id = id;
        } else if (code == "ce" && !args.empty()) {
            int v = 0;
            if (parse_int(args[0], v)) e.ce = v;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Data passed from the protocol layer to the search
// ---------------------------------------------------------------------------

// "position ..." exactly as the GUI sent it. The board is built from this
// later: set up `fen`, then play `moves` in order.
struct PositionCmd {
    std::string fen{StartFen};
    std::vector<std::string> moves;
    chess::Board board{StartFen};  // fen with all of `moves` played

    bool white_to_move() const { return board.sideToMove() == chess::Color::WHITE; }
};

struct SearchLimits {
    int depth = 0;              // 0 = no depth limit
    TimeMs wtime = -1;          // -1 = not given
    TimeMs btime = -1;
    TimeMs winc = 0;
    TimeMs binc = 0;
    int movestogo = 0;          // 0 = sudden death / not given
    TimeMs movetime = 0;        // 0 = not given
    std::uint64_t nodes = 0;    // 0 = no node limit
    int mate = 0;               // search for mate in N moves, 0 = off
    int perft = 0;              // non-standard: perft N
    bool infinite = false;
    bool ponder = false;
    std::vector<std::string> searchmoves;  // empty = all legal moves
};

struct Options {
    int hash_mb = 16;
    int threads = 1;
    int multipv = 1;
    int move_overhead = 30;     // ms held back per move for GUI / OS lag
    bool ponder = false;
    bool own_book = false;       // play from the built-in opening book (off by default: tournament play)
};

// Soft limit: don't start a new iteration after this.
// Hard limit: abort the current iteration immediately.
// -1 means no limit.
struct TimeBudget {
    TimeMs soft = -1;
    TimeMs hard = -1;
    bool clock = false;  // from the game clock (not movetime): soft may be scaled
};

// Time management (SR_TM): the soft limit is scaled by how settled the
// search looks: less time while the best move stays the same from one
// iteration to the next, more when it just changed or the score is falling.
// Tested on a real clock (tools/match "c<base>+<inc>") vs. the fixed budget:
// 300 games @2s+20ms +9; 600 @2s+20ms about +12; 600 @4s+40ms about +24
// (1200 games about +18, LOS 99%). On.
#ifndef SR_TM
#define SR_TM 1
#endif
inline constexpr bool UseTimeManagement = SR_TM;

inline TimeBudget allocate_time(const SearchLimits& lim, bool white, int overhead) {
    TimeBudget tb;
    if (lim.infinite) return tb;

    if (lim.movetime > 0) {
        tb.soft = tb.hard = std::max<TimeMs>(1, lim.movetime - overhead);
        return tb;
    }

    const TimeMs time = white ? lim.wtime : lim.btime;
    const TimeMs inc  = white ? lim.winc  : lim.binc;
    if (time < 0) return tb;  // no clock given, e.g. "go depth 10"

    const TimeMs usable = std::max<TimeMs>(1, time - overhead);
    const TimeMs mtg = lim.movestogo > 0 ? std::min(lim.movestogo, 50) : 30;
    const TimeMs cap = std::max<TimeMs>(1, usable * 9 / 10);

    tb.soft = std::clamp<TimeMs>(usable / mtg + inc * 3 / 4, 1, cap);
    tb.hard = std::clamp<TimeMs>(tb.soft * 3, 1, cap);
    tb.clock = true;
    return tb;
}

// Everything the search needs. The real search should call should_stop()
// every few thousand nodes and report with the info_* helpers.
struct SearchContext {
    PositionCmd position;
    SearchLimits limits;
    Options options;
    TimeBudget budget;

    std::atomic<bool>& stop;       // set by "stop" / "quit" / new command
    std::atomic<bool>& pondering;  // true until "ponderhit"
    std::atomic<TimeMs>& start;    // reset on "ponderhit"
    search::TranspositionTable& tt;
    search::OrderTables* tables;   // move-ordering stats kept between moves (nullptr = fresh each search)
    // Helper threads' tables, kept between moves when HelperTables is on.
    std::vector<std::unique_ptr<search::OrderTables>>* helper_tables = nullptr;

    TimeMs elapsed() const { return now_ms() - start.load(); }

    // SR_TM state, updated from each completed line 1.
    mutable std::uint16_t tm_best = 0;
    mutable int tm_stable = 0;      // iterations the best move has stayed the same
    mutable int tm_prev_score = 0;
    mutable int tm_depth = 0;
    mutable double tm_scale = 1.0;
    void tm_update(int depth, std::uint16_t best, int score) const {
        if (depth <= tm_depth) return;
        tm_stable = (tm_depth > 0 && best == tm_best) ? tm_stable + 1 : 0;
        const int drop = tm_depth > 0 ? tm_prev_score - score : 0;
        tm_best = best;
        tm_prev_score = score;
        tm_depth = depth;
        const double stab = std::clamp(1.3 - 0.12 * tm_stable, 0.6, 1.3);
        const double fall = std::clamp(1.0 + drop / 100.0, 1.0, 1.5);
        tm_scale = depth >= 4 ? stab * fall : 1.0;
    }

    // Hard stop: GUI said stop, node limit hit, or hard time limit passed.
    bool should_stop(std::uint64_t nodes) const {
        if (stop.load(std::memory_order_relaxed)) return true;
        if (limits.nodes && nodes >= limits.nodes) return true;
        if (!pondering.load(std::memory_order_relaxed) && budget.hard >= 0 &&
            elapsed() >= budget.hard)
            return true;
        return false;
    }

    // Soft stop: checked between iterative-deepening iterations.
    bool should_start_next_iteration(int next_depth) const {
        if (stop.load()) return false;
        if (next_depth > MaxDepth) return false;
        if (limits.depth > 0 && next_depth > limits.depth) return false;
        if (!pondering.load() && budget.soft >= 0) {
            TimeMs soft = budget.soft;
            if (UseTimeManagement && budget.clock)
                soft = std::min<TimeMs>(budget.hard, static_cast<TimeMs>(static_cast<double>(soft) * tm_scale));
            if (elapsed() >= soft) return false;
        }
        return true;
    }

    // Number of PV lines to report (capped by searchmoves when given; the real
    // search should also cap it by the number of legal moves).
    int multipv() const {
        int n = options.multipv;
        if (!limits.searchmoves.empty())
            n = std::min<int>(n, static_cast<int>(limits.searchmoves.size()));
        return std::max(1, n);
    }

    // score_cp is from the side to move's view. mate_in != 0 prints
    // "score mate N" instead (negative = getting mated).
    void info_pv(int depth, int seldepth, int pv_index, int score_cp, int mate_in,
                 std::uint64_t nodes, const std::vector<std::string>& pv,
                 int hashfull = -1) const {
        const TimeMs t = std::max<TimeMs>(1, elapsed());
        std::ostringstream os;
        os << "info depth " << depth << " seldepth " << seldepth
           << " multipv " << pv_index;
        if (mate_in != 0) os << " score mate " << mate_in;
        else              os << " score cp " << score_cp;
        os << " nodes " << nodes << " nps " << (nodes * 1000 / static_cast<std::uint64_t>(t))
           << " time " << t;
        if (hashfull >= 0) os << " hashfull " << hashfull;
        if (!pv.empty()) {
            os << " pv";
            for (const auto& m : pv) os << ' ' << m;
        }
        send(os.str());
    }
};

// ---------------------------------------------------------------------------
// Search placeholder. Replace the body with the real search.
// Must return the best move in UCI notation (and optionally a ponder move).
// ---------------------------------------------------------------------------
struct SearchResult {
    std::string best = "0000";
    std::string ponder;  // empty = none
};

// Runs the PVS search from Search.h under the limits in ctx.
inline SearchResult run_search(SearchContext& ctx) {
    const chess::Board& board = ctx.position.board;
    SearchResult result;

    // UCI searchmoves -> legal chess::Move list (empty = all moves).
    std::vector<chess::Move> allowed;
    for (const auto& text : ctx.limits.searchmoves) {
        const chess::Move m = find_legal_move(board, text);
        if (m != chess::Move::NO_MOVE) allowed.push_back(m);
        else send("info string searchmoves: ignoring illegal move ", text);
    }

    std::atomic<bool> helpers_stop{false};
    std::atomic<std::uint64_t> helper_nodes{0};

    search::Hooks hooks;
    hooks.should_stop = [&](std::uint64_t nodes) { return ctx.should_stop(nodes); };
    hooks.start_next_iteration = [&](int depth) { return ctx.should_start_next_iteration(depth); };
    hooks.report = [&](int depth, int seldepth, int multipv, const search::Line& line,
                       std::uint64_t nodes, int hashfull) {
        std::vector<std::string> pv;
        for (const auto& m : line.pv) pv.push_back(chess::uci::moveToUci(m, board.chess960()));
        const int mate = search::is_mate_score(line.score) ? search::mate_in_moves(line.score) : 0;
        ctx.info_pv(depth, seldepth, multipv, line.score, mate, nodes + helper_nodes.load(), pv, hashfull);
        if (multipv == 1 && !line.pv.empty()) ctx.tm_update(depth, line.pv.front().move(), line.score);
    };

    // Lazy SMP helpers (Threads > 1): same position, shared TT, stopped when
    // the main search finishes. Only the main thread reports and picks the move.
    // A/B, 25 ms/move: 2 threads vs 1 thread +99 Elo (600 games, 95% +76..+122);
    // 1 thread vs. the engine before the lock-free TT: -2 (600), i.e. no cost.
    std::vector<std::unique_ptr<search::Searcher>> helpers;
    for (int i = 1; i < std::max(1, ctx.options.threads); ++i) {
        search::Hooks h;
        h.should_stop = [&](std::uint64_t) {
            helper_nodes.fetch_add(2048, std::memory_order_relaxed);  // polled every 2048 nodes
            return helpers_stop.load(std::memory_order_relaxed) || ctx.stop.load(std::memory_order_relaxed);
        };
        h.start_next_iteration = [](int depth) { return depth < search::MaxPly - 1; };
        h.report = [](int, int, int, const search::Line&, std::uint64_t, int) {};
        search::OrderTables* ht = nullptr;
        if (search::HelperTables && ctx.helper_tables) {
            auto& v = *ctx.helper_tables;
            while (static_cast<int>(v.size()) < i) v.push_back(std::make_unique<search::OrderTables>());
            ht = v[static_cast<std::size_t>(i - 1)].get();
        }
        helpers.push_back(std::make_unique<search::Searcher>(board, ctx.tt, std::move(h), ht));
        helpers.back()->set_thread_id(i);
    }
    std::vector<std::thread> pool;
    for (auto& helper : helpers) {
        try {
            pool.emplace_back([&allowed, s = helper.get()] { s->iterate(allowed, 1); });
        } catch (const std::system_error&) {
            // The OS wouldn't create another thread: search with the ones we have.
            send("info string could only start ", pool.size() + 1, " search threads");
            break;
        }
    }

    // Heap-allocated: the searcher holds a PV table too big for comfort on a
    // thread stack.
    auto searcher = std::make_unique<search::Searcher>(board, ctx.tt, std::move(hooks), ctx.tables);
    const search::Result r = searcher->iterate(allowed, ctx.multipv());
    helpers_stop.store(true);
    for (auto& t : pool) t.join();

    if (r.best == chess::Move::NO_MOVE) {
        // No legal moves: checkmate ("score mate 0") or stalemate.
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        if (legal.empty() && board.inCheck()) send("info depth 0 score mate 0");
        else                                  send("info depth 0 score cp 0");
        return result;  // bestmove 0000
    }

    result.best = chess::uci::moveToUci(r.best, board.chess960());
    if (r.ponder != chess::Move::NO_MOVE)
        result.ponder = chess::uci::moveToUci(r.ponder, board.chess960());
    return result;
}

// ---------------------------------------------------------------------------
// Protocol handler
// ---------------------------------------------------------------------------
class Uci {
public:
    Uci() { eval::Kpk::get(); }  // build the KPK table now, not during the first search
    ~Uci() { stop_search(); }
    Uci(const Uci&) = delete;
    Uci& operator=(const Uci&) = delete;

    // Read commands from stdin until "quit" or EOF.
    void loop() {
        std::string line;
        while (std::getline(std::cin, line))
            if (!handle(line)) return;
        stop_search();
    }

    // Process one command line. Returns false on "quit".
    bool handle(std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream is(line);
        std::string cmd;
        if (!(is >> cmd)) return true;

        if      (cmd == "uci")        cmd_uci();
        else if (cmd == "isready")    send("readyok");
        else if (cmd == "ucinewgame") { stop_search(); position_ = {}; tt_.clear(); tables_ = std::make_unique<search::OrderTables>(); helper_tables_.clear(); }
        else if (cmd == "setoption")  cmd_setoption(is);
        else if (cmd == "position")   { stop_search(); cmd_position(is); }
        else if (cmd == "go")         cmd_go(is);
        else if (cmd == "stop")       stop_search();
        else if (cmd == "ponderhit")  cmd_ponderhit();
        else if (cmd == "quit")       { stop_search(); return false; }
        else if (cmd == "d")          cmd_display();
        else if (cmd == "eval")       cmd_eval();
        else if (cmd == "book")       cmd_book();
        else if (cmd == "bench")      cmd_bench(is);
        else if (cmd == "orderstats") cmd_orderstats(is);
        else if (cmd == "testsuite")  cmd_testsuite(is);
        else if (cmd == "debug" || cmd == "register") { /* accepted, ignored */ }
        else send("info string unknown command: ", line);
        return true;
    }

    const PositionCmd& position() const { return position_; }
    const Options& options() const { return options_; }

private:
    // ---- uci ------------------------------------------------------------
    void cmd_uci() {
        send("id name ", EngineName);
        send("id author ", EngineAuthor);
        send("option name Hash type spin default 16 min 1 max 65536");
        send("option name Threads type spin default 1 min 1 max 1024");
        send("option name MultiPV type spin default 1 min 1 max 256");
        send("option name Move Overhead type spin default 30 min 0 max 5000");
        send("option name Ponder type check default false");
        send("option name OwnBook type check default false");  // off for play; datagen / SPRT / SPSA use their own openings
        send("option name BookMode type combo default ", book::ModeNames[BOOK_MODE],
             " var Wide var Normal var Tight var Optimal");
        if (SP_TUNE)
            for (const auto& t : search::tune_params())
                send("option name ", t.name, " type spin default ", t.def, " min ", t.min, " max ", t.max);
        send("uciok");
    }

    // ---- setoption name <id> [value <x>] ---------------------------------
    // Option names can contain spaces ("Move Overhead"), so everything
    // between "name" and "value" is the name.
    void cmd_setoption(std::istringstream& is) {
        std::string tok, name, value;
        bool in_value = false;
        is >> tok;  // "name"
        while (is >> tok) {
            if (!in_value && tok == "value") { in_value = true; continue; }
            std::string& dst = in_value ? value : name;
            if (!dst.empty()) dst += ' ';
            dst += tok;
        }

        const std::string key = to_lower(name);
        auto spin = [&](int& target, int lo, int hi) {
            int v = 0;
            if (!parse_int(value, v)) {
                send("info string invalid value for ", name, ": ", value);
                return;
            }
            target = std::clamp(v, lo, hi);
        };

        if (key == "hash") {
            stop_search();  // never resize under a running search
            spin(options_.hash_mb, 1, 65536);
            tt_.resize(static_cast<std::size_t>(options_.hash_mb));
        }
        else if (key == "threads") {
            spin(options_.threads, 1, 1024);
            // More threads than logical cores only makes them compete for the
            // same cores (slower, not stronger), so cap at the hardware count.
            const unsigned hw = std::thread::hardware_concurrency();  // 0 = unknown
            if (hw > 0 && options_.threads > static_cast<int>(hw)) {
                send("info string Threads ", options_.threads, " is more than this machine's ", hw,
                     " logical cores; using ", hw);
                options_.threads = static_cast<int>(hw);
            }
        }
        else if (key == "multipv")       spin(options_.multipv, 1, 256);
        else if (key == "move overhead") spin(options_.move_overhead, 0, 5000);
        else if (key == "ponder")        options_.ponder = (to_lower(value) == "true");
        else if (key == "ownbook")       options_.own_book = (to_lower(value) == "true");
        else if (key == "bookmode") {
            bool found = false;
            for (int i = 0; i < 4; ++i)
                if (to_lower(value) == to_lower(book::ModeNames[i])) {
                    book::Book::instance().set_mode(static_cast<book::Mode>(i));
                    found = true;
                }
            if (!found) send("info string unknown BookMode: ", value);
        }
        else if (SP_TUNE && set_tune_param(key, value)) {}
        else send("info string unknown option: ", name);
    }

    // SP_TUNE builds: search parameters as spin options (names case-insensitive).
    bool set_tune_param(const std::string& key, const std::string& value) {
        for (auto& t : search::tune_params())
            if (to_lower(t.name) == key) {
                *t.value = std::clamp(std::atoi(value.c_str()), t.min, t.max);
                search::tune_apply();
                return true;
            }
        return false;
    }

    // ---- position [startpos | fen <fen>] [moves <m1> ... <mN>] ------------
    void cmd_position(std::istringstream& is) {
        PositionCmd p;
        std::string tok;
        if (!(is >> tok)) { send("info string position: missing arguments"); return; }

        bool has_moves = false;
        if (tok == "startpos") {
            p.fen = StartFen;
            if (is >> tok) has_moves = (tok == "moves");
        } else if (tok == "fen") {
            std::vector<std::string> fields;
            while (is >> tok) {
                if (tok == "moves") { has_moves = true; break; }
                fields.push_back(tok);
            }
            // Board, side, castling, en passant are required; some GUIs omit
            // the halfmove and fullmove counters.
            if (fields.size() < 4 || fields.size() > 6) {
                send("info string position: bad fen");
                return;
            }
            if (fields.size() < 5) fields.push_back("0");
            if (fields.size() < 6) fields.push_back("1");
            p.fen.clear();
            for (const auto& f : fields) {
                if (!p.fen.empty()) p.fen += ' ';
                p.fen += f;
            }
        } else {
            send("info string position: expected startpos or fen");
            return;
        }

        p.board.setFen(p.fen);
        if (!board_is_sane(p.board)) {
            send("info string position: illegal fen ", p.fen);
            return;
        }

        if (has_moves) {
            while (is >> tok) {
                const chess::Move m = is_move_token(tok) ? find_legal_move(p.board, tok)
                                                         : chess::Move(chess::Move::NO_MOVE);
                if (m == chess::Move::NO_MOVE) {
                    send("info string position: illegal move '", tok, "', ignoring the rest");
                    break;
                }
                p.board.makeMove(m);
                p.moves.push_back(tok);
            }
        }
        position_ = std::move(p);
    }

    // ---- go ... -----------------------------------------------------------
    void cmd_go(std::istringstream& is) {
        stop_search();  // GUI should have sent stop already; be safe

        std::vector<std::string> t;
        for (std::string tok; is >> tok;) t.push_back(tok);

        SearchLimits lim;
        for (std::size_t i = 0; i < t.size(); ++i) {
            const std::string& k = t[i];
            auto next = [&](auto& out) {
                if (i + 1 < t.size() && parse_int(t[i + 1], out)) ++i;
                else send("info string go: bad or missing value for ", k);
            };

            if      (k == "depth")     next(lim.depth);
            else if (k == "wtime")     next(lim.wtime);
            else if (k == "btime")     next(lim.btime);
            else if (k == "winc")      next(lim.winc);
            else if (k == "binc")      next(lim.binc);
            else if (k == "movestogo") next(lim.movestogo);
            else if (k == "movetime")  next(lim.movetime);
            else if (k == "nodes")     next(lim.nodes);
            else if (k == "mate")      next(lim.mate);
            else if (k == "perft")     next(lim.perft);
            else if (k == "infinite")  lim.infinite = true;
            else if (k == "ponder")    lim.ponder = true;
            else if (k == "searchmoves") {
                while (i + 1 < t.size() && is_move_token(t[i + 1]))
                    lim.searchmoves.push_back(t[++i]);
            }
            else send("info string go: unknown parameter ", k);
        }

        if (lim.perft > 0) {
            cmd_perft(lim.perft);
            return;
        }

        // Opening book: answer instantly with a book move when there is one.
        // Not for analysis-style searches (infinite, ponder, searchmoves,
        // mate) where the GUI wants the engine's own evaluation.
        if (options_.own_book && !lim.infinite && !lim.ponder && lim.searchmoves.empty() &&
            lim.mate == 0) {
            auto& bk = book::Book::instance();
            const chess::Move m = bk.probe(position_.board);
            if (m != chess::Move::NO_MOVE) {
                const std::string uci = chess::uci::moveToUci(m, position_.board.chess960());
                int cp = 0;  // cached build-time score, side to move's view
                for (const auto& c : bk.moves(position_.board))
                    if (c.move == m) cp = position_.board.sideToMove() == chess::Color::WHITE ? c.score : -c.score;
                send("info string book move ", uci, " (", book::ModeNames[static_cast<int>(bk.mode())], ")");
                send("info depth 0 score cp ", cp, " nodes 0 time 0 pv ", uci);
                send("bestmove ", uci);
                return;
            }
        }

        start_search(std::move(lim));
    }

    // "book": book moves for the current position with their weights and
    // cached scores; "*" marks moves the current BookMode may play.
    void cmd_book() {
        auto& bk = book::Book::instance();
        const auto moves = bk.moves(position_.board);
        const auto ok = bk.playable(position_.board, bk.mode());
        if (moves.empty()) {
            send("info string out of book (", book::Book::instance().positions(), " book positions)");
            return;
        }
        for (const auto& m : moves)
            send("info string book ", chess::uci::moveToSan(position_.board, m.move), " (",
                 chess::uci::moveToUci(m.move, position_.board.chess960()), ")  weight ", m.weight,
                 m.wide_weight ? "  wide " + std::to_string(m.wide_weight) : std::string(),
                 "  score ", m.score, " (white)",
                 std::any_of(ok.begin(), ok.end(), [&](const book::BookMove& c) { return c.move == m.move; }) ? "  *" : "");
        send("info string BookMode ", book::ModeNames[static_cast<int>(bk.mode())]);
    }

    // "go perft N": per-move counts, then the total (same format as Stockfish).
    void cmd_perft(int depth) {
        chess::Board board = position_.board;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        const TimeMs t0 = now_ms();
        std::uint64_t total = 0;
        for (const auto& m : moves) {
            board.makeMove(m);
            const std::uint64_t n = depth > 1 ? search::perft(board, depth - 1) : 1;
            board.unmakeMove(m);
            total += n;
            send(chess::uci::moveToUci(m, board.chess960()), ": ", n);
        }
        const TimeMs ms = std::max<TimeMs>(1, now_ms() - t0);
        send("");
        send("Nodes searched: ", total);
        send("info string perft ", depth, " time ", ms, " ms nps ", total * 1000 / static_cast<std::uint64_t>(ms));
    }

    void cmd_ponderhit() {
        {
            std::lock_guard lock(wait_mutex_);
            start_ms_ = now_ms();  // our clock starts now
            pondering_ = false;
        }
        wait_cv_.notify_all();
    }

    void cmd_display() {
        send("info string fen ", position_.fen);
        send("info string current fen ", position_.board.getFen());
        std::string moves;
        for (const auto& m : position_.moves) moves += ' ' + m;
        send("info string moves", moves.empty() ? " (none)" : moves);
        send("info string side to move ", position_.white_to_move() ? "white" : "black");
        send("info string multipv ", options_.multipv, " hash ", options_.hash_mb,
             " threads ", options_.threads, " overhead ", options_.move_overhead);
    }

    // Evaluation breakdown for the current position: the opening and endgame
    // piece-square bonus of every piece (from its owner's point of view),
    // then each term and the phase blend.
    void cmd_eval() {
        const eval::EvalBoard board(position_.board);
        const eval::Terms t = board.terms();
        if (eval::UseNnue)
            send("info string NNUE eval (side to move, used by the search): ", eval::evaluate(board),
                 " cp; the hand-written eval below is for reference");

        auto grid = [&](const char* title, auto lookup) {
            send("info string ", title);
            send("info string      a     b     c     d     e     f     g     h");
            for (int rank = 7; rank >= 0; --rank) {
                std::ostringstream row;
                row << "info string " << (rank + 1) << ' ';
                for (int file = 0; file < 8; ++file) {
                    const chess::Square sq(rank * 8 + file);
                    const chess::Piece p = board.at(sq);
                    if (p == chess::Piece::NONE) { row << "    . "; continue; }
                    const int v = lookup(p, sq);  // owner's point of view
                    const char letter = "PNBRQKpnbrqk"[static_cast<int>(p.internal())];
                    char buf[16];
                    std::snprintf(buf, sizeof buf, " %c%+4d", letter, v);
                    row << buf;
                }
                send(row.str());
            }
        };
        grid("PST opening:", [](chess::Piece p, chess::Square sq) { return eval::pst_mg_of(p, sq); });
        grid("PST endgame:", [](chess::Piece p, chess::Square sq) { return eval::pst_eg_of(p, sq); });

        // Per side, each from its own point of view.
        const eval::Positional pos = eval::positional(board);
        auto row = [](const std::string& name, int wmg, int weg, int bmg, int beg) {
            char buf[112];
            std::snprintf(buf, sizeof buf, "%-12s %7d %7d   %7d %7d", name.c_str(), wmg, weg, bmg, beg);
            send("info string ", buf);
        };
        const auto& w = t.side[0];
        const auto& b = t.side[1];
        send("info string                   white           black      (each side's own view)");
        send("info string term           mg      eg        mg      eg");
        row("material", w.material_mg, w.material_eg, b.material_mg, b.material_eg);
        row("pst", w.pst_mg, w.pst_eg, b.pst_mg, b.pst_eg);
        row("mobility", w.mobility_mg + pos.ai.slider[0].mg, w.mobility_eg + pos.ai.slider[0].eg,
            b.mobility_mg + pos.ai.slider[1].mg, b.mobility_eg + pos.ai.slider[1].eg);
        for (int i = 0; i < eval::TermCount; ++i) {
            const auto& pw = pos.side[0].t[i];
            const auto& pb = pos.side[1].t[i];
            if (pw.mg || pw.eg || pb.mg || pb.eg) row(eval::TermNames[i], pw.mg, pw.eg, pb.mg, pb.eg);
        }
        const eval::MgEg wt = eval::side_totals(w, pos.ai.slider[0]);
        const eval::MgEg bt = eval::side_totals(b, pos.ai.slider[1]);
        const eval::MgEg wp = pos.side[0].total(), bp = pos.side[1].total();
        row("total", wt.mg + wp.mg, wt.eg + wp.eg, bt.mg + bp.mg, bt.eg + bp.eg);

        const int phase = t.phase();
        const int mg = wt.mg - bt.mg + pos.score.mg;
        int eg = wt.eg - bt.eg + pos.score.eg;
        const int scale = eval::endgame_scale(pos.ai, eg);
        eg = eg * scale / 64;
        int blended = eval::taper(mg, eg, phase);
        const bool white = board.sideToMove() == chess::Color::WHITE;
        send("info string phase        ", phase, "/", eval::PhaseMax, " (non-pawn material ", w.npm + b.npm,
             "; ", phase * 100 / eval::PhaseMax, "% middlegame), endgame scale ", scale, "/64");
        if (eval::EvalUseTempo) blended += white ? eval::Tempo : -eval::Tempo;
        send("info string blended      ", blended, " (white's view, incl. tempo), halfmove clock ",
             board.halfMoveClock(), " -> ", eval::fifty_move_scale(blended, static_cast<int>(board.halfMoveClock())),
             ", eval ", eval::evaluate(board), " (", white ? "white" : "black", " to move)");
    }

    // "bench [depth] [hashMB]": searches every BenchFens position to a fixed depth with
    // a fresh 16 MB hash (or hashMB) each, independent of the Hash option, so the node
    // count only changes when the search itself does. Runs on this thread.
    void cmd_bench(std::istringstream& is) {
        stop_search();

        int depth = BenchDepth;
        std::string tok;
        if (is >> tok && !parse_int(tok, depth)) {
            send("info string bench: bad depth '", tok, "'");
            return;
        }
        depth = std::clamp(depth, 1, search::MaxPly - 1);
        int hash_mb = 16;  // optional second argument: hash size in MB (TT pressure tests)
        if (is >> tok && (!parse_int(tok, hash_mb) || hash_mb < 1)) {
            send("info string bench: bad hash size '", tok, "'");
            return;
        }

        search::TranspositionTable tt(static_cast<std::size_t>(hash_mb));
        std::uint64_t total = 0;
        const int count = static_cast<int>(std::size(BenchFens));
        const TimeMs t0 = now_ms();

        for (int i = 0; i < count; ++i) {
            tt.clear();
            const chess::Board board(BenchFens[i]);

            search::Hooks hooks;
            hooks.should_stop = [](std::uint64_t) { return false; };
            hooks.start_next_iteration = [depth](int d) { return d <= depth; };
            hooks.report = [](int, int, int, const search::Line&, std::uint64_t, int) {};

            auto searcher = std::make_unique<search::Searcher>(board, tt, std::move(hooks));
            const search::Result r = searcher->iterate({}, 1);
            total += searcher->nodes();

            send("info string bench ", i + 1, "/", count, "  nodes ", searcher->nodes(),
                 "  bestmove ", chess::uci::moveToUci(r.best, board.chess960()),
                 "  fen ", BenchFens[i]);
        }

        const TimeMs ms = std::max<TimeMs>(1, now_ms() - t0);
        const std::uint64_t nps = total * 1000 / static_cast<std::uint64_t>(ms);
        send("===========================");
        send("Depth           : ", depth);
        send("Total time (ms) : ", ms);
        send("Nodes searched  : ", total);
        send("Nodes/second    : ", nps);
        send(total, " nodes ", nps, " nps");
    }

    // "orderstats [depth] [hashMB]": bench (default depth 13, 16 MB) with search statistics:
    // where the cutoff move was in the move list (overall and by kind of
    // move), and how often reductions / pruning fired or had to be undone.
    // Works with any parameter values (SP_TUNE builds: setoption first), so
    // move-ordering and pruning changes can be compared without games.
    void cmd_orderstats(std::istringstream& is) {
        std::string rest, tok;
        if (!(is >> tok)) tok = "13";
        if (is >> rest) tok += " " + rest;  // optional hash size in MB
        std::istringstream bench_args(tok);
        search::g_stats = {};
        search::tt_stats = {};
        search::g_collect_stats = search::tt_collect_stats = true;
        cmd_bench(bench_args);
        search::g_collect_stats = search::tt_collect_stats = false;
        const search::SearchStats& st = search::g_stats;
        auto pct = [](long a, long b) { return b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0; };
        char buf[512];
        auto line = [&](const char* fmt, auto... args) {
            std::snprintf(buf, sizeof buf, fmt, args...);
            send(buf);
        };
        send("=========================== search statistics");
        line("nodes            : %ld main search, %ld qsearch (%.1f%% qsearch)", st.nodes_main, st.nodes_qs,
             pct(st.nodes_qs, st.nodes_main + st.nodes_qs));
        line("cutoffs          : %ld  first move %.1f%%  (non-PV nodes %.1f%%)  within 3 %.1f%%  mean position %.3f",
             st.cutoffs, pct(st.first, st.cutoffs), pct(st.nonpv_first, st.nonpv), pct(st.top3, st.cutoffs),
             st.cutoffs ? static_cast<double>(st.idx_sum) / static_cast<double>(st.cutoffs) : 0.0);
        const char* names[8] = {"TT move", "good capture", "killer 1", "killer 2", "counter", "quiet (history)", "bad capture", "under-promotion"};
        send("by kind of cutting move: share of cutoffs, mean position, share of late cutoffs (position > 1), kind of the failed first move");
        for (int c = 0; c < 8; ++c)
            line("  %-16s %5.1f%%  pos %6.2f  late %5.1f%%  failed-first %5.1f%%", names[c], pct(st.cat[c], st.cutoffs),
                 st.cat[c] ? static_cast<double>(st.cat_idx[c]) / static_cast<double>(st.cat[c]) : 0.0,
                 pct(st.late_cat[c], st.late), pct(st.fail_first_cat[c], st.late));
        line("LMR              : %ld reduced searches (%.0f per 1000 nodes), %.2f%% re-searched (beat alpha)", st.lmr,
             1000.0 * static_cast<double>(st.lmr) / static_cast<double>(std::max(1L, st.nodes_main + st.nodes_qs)), pct(st.lmr_research, st.lmr));
        line("null move        : %ld tried, %.1f%% cut off", st.nmp, pct(st.nmp_cut, st.nmp));
        line("ProbCut          : %ld tried, %.1f%% cut off", st.pc, pct(st.pc_cut, st.pc));
        line("singular         : %ld searches, %.1f%% extended, %.1f%% multi-cut", st.se, pct(st.se_ext, st.se), pct(st.se_multicut, st.se));
        if (st.iid) line("IID              : %ld searches, %.1f%% found a move", st.iid, pct(st.iid_move, st.iid));
        if (st.nnue_ord) line("NNUE ordering    : %ld nodes, %ld evals (%.1f per 1000 nodes)", st.nnue_ord, st.nnue_evals,
                              1000.0 * static_cast<double>(st.nnue_evals) / static_cast<double>(std::max(1L, st.nodes_main + st.nodes_qs)));
        line("node cutoffs     : reverse futility %ld, razoring %ld (per 1000 main nodes: %.1f, %.1f)", st.rfp, st.razor,
             1000.0 * static_cast<double>(st.rfp) / static_cast<double>(std::max(1L, st.nodes_main)),
             1000.0 * static_cast<double>(st.razor) / static_cast<double>(std::max(1L, st.nodes_main)));
        line("moves pruned     : late-move %ld, futility %ld, SEE quiet %ld, history %ld, SEE capture %ld", st.lmp, st.fut, st.see_q, st.hist, st.see_c);
        const search::TTStats& ts = search::tt_stats;
        line("TT (main search) : %ld probes, found %.1f%% (with a bound %.1f%%, with a move %.1f%%)", st.tt_probes,
             pct(st.tt_found, st.tt_probes), pct(st.tt_hit, st.tt_probes), pct(st.tt_move, st.tt_probes));
        line("TT stores        : %ld, same key %.1f%%, empty slot %.1f%%, over another key %.1f%%, dropped %.1f%%  (%s)",
             ts.stores, pct(ts.same_key, ts.stores), pct(ts.filled_empty, ts.stores), pct(ts.replaced_other, ts.stores),
             pct(ts.dropped, ts.stores), search::TtBuckets ? "buckets of 4" : "one slot per key");
        send("reference: Stockfish dev (Oct 2026), its bench at depth 15 (7.2M nodes), counted the same way: first move 87.9%,");
        send("  within 3 97.1%, mean position 1.28; LMR re-searched 3.2% (about 420 reduced searches per 1000 nodes);");
        send("  null move cut off 53.5%; singular extended 38.1%");
        send("ORDERSTATS-SUMMARY first=", pct(st.first, st.cutoffs), " top3=", pct(st.top3, st.cutoffs), " meanpos=",
             st.cutoffs ? static_cast<double>(st.idx_sum) / static_cast<double>(st.cutoffs) : 0.0,
             " quietpos=", st.cat[5] ? static_cast<double>(st.cat_idx[5]) / static_cast<double>(st.cat[5]) : 0.0,
             " lmr_research=", pct(st.lmr_research, st.lmr), " nodes=", st.nodes_main + st.nodes_qs);
    }

    // "testsuite <file.epd> [ms]": searches every position for `ms` ms
    // (default 1000) and checks the move against its bm / am operations.
    // A position counts as solved if the final move is a bm (and not an am).
    // "Found" is when the search first settled on a correct move for good.
    // Uses a fresh hash of the current Hash size per position. Runs on this
    // thread, so it can't be interrupted with "stop".
    void cmd_testsuite(std::istringstream& is) {
        stop_search();

        std::string path, tok;
        if (!(is >> path)) { send("usage: testsuite <file.epd> [ms per position]"); return; }
        int ms = 1000;
        if (is >> tok && (!parse_int(tok, ms) || ms < 1)) { send("testsuite: bad time '", tok, "'"); return; }

        std::ifstream in(path);
        if (!in) { send("testsuite: cannot open '", path, "'"); return; }

        std::vector<EpdEntry> suite;
        for (std::string line; std::getline(in, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            EpdEntry e;
            if (line.empty() || line[0] == '#' || !parse_epd(line, e)) continue;
            if (e.id.empty()) e.id = "#" + std::to_string(suite.size() + 1);
            suite.push_back(std::move(e));
        }
        if (suite.empty()) { send("testsuite: no positions in '", path, "'"); return; }

        search::TranspositionTable tt(static_cast<std::size_t>(options_.hash_mb));
        int solved = 0, skipped = 0, ce_count = 0;
        TimeMs found_time_sum = 0;
        std::uint64_t total_nodes = 0;
        long long ce_abs_sum = 0;
        std::vector<std::string> failed;

        for (const auto& e : suite) {
            const chess::Board board(e.fen);
            if (!board_is_sane(board)) { send(e.id, "  SKIP  illegal position"); ++skipped; continue; }

            // Resolve the SAN lists against this position.
            auto resolve = [&](const std::vector<std::string>& sans, std::vector<chess::Move>& out) {
                for (const auto& san : sans) {
                    const chess::Move m = find_san_move(board, san);
                    if (m == chess::Move::NO_MOVE) return false;
                    out.push_back(m);
                }
                return true;
            };
            std::vector<chess::Move> bm, am;
            if (!resolve(e.bm, bm) || !resolve(e.am, am) || (bm.empty() && am.empty())) {
                send(e.id, "  SKIP  can't read bm/am");
                ++skipped;
                continue;
            }
            auto correct = [&](const chess::Move& m) {
                const bool in_bm = std::find(bm.begin(), bm.end(), m) != bm.end();
                const bool in_am = std::find(am.begin(), am.end(), m) != am.end();
                return (bm.empty() || in_bm) && !in_am;
            };

            tt.clear();
            const TimeMs t0 = now_ms();
            TimeMs found_at = -1;
            int found_depth = 0, last_depth = 0;
            eval::Score last_score = 0;

            search::Hooks hooks;
            hooks.should_stop = [&](std::uint64_t) { return now_ms() - t0 >= ms; };
            hooks.start_next_iteration = [&](int) { return now_ms() - t0 < ms; };
            hooks.report = [&](int depth, int, int multipv, const search::Line& line,
                               std::uint64_t, int) {
                if (multipv != 1) return;
                last_depth = depth;
                last_score = line.score;
                if (correct(line.pv.front())) {
                    if (found_at < 0) { found_at = now_ms() - t0; found_depth = depth; }
                } else {
                    found_at = -1;  // switched away: not found (yet)
                }
            };

            auto searcher = std::make_unique<search::Searcher>(board, tt, std::move(hooks));
            const search::Result r = searcher->iterate({}, 1);
            total_nodes += searcher->nodes();

            const bool ok = correct(r.best);
            std::string expected;
            for (const auto& b : e.bm) expected += (expected.empty() ? "" : " ") + b;
            if (!e.am.empty()) {
                expected += expected.empty() ? "not" : ", not";
                for (const auto& a : e.am) expected += " " + a;
            }

            char line[256];
            if (ok) {
                ++solved;
                found_time_sum += std::max<TimeMs>(found_at, 0);
                std::snprintf(line, sizeof line, "%-10s ok    %-8s found %5lld ms, depth %2d",
                              e.id.c_str(), chess::uci::moveToSan(board, r.best).c_str(),
                              static_cast<long long>(std::max<TimeMs>(found_at, 0)), found_depth);
            } else {
                failed.push_back(e.id);
                std::snprintf(line, sizeof line, "%-10s FAIL  %-8s expected %s (depth %d)",
                              e.id.c_str(), chess::uci::moveToSan(board, r.best).c_str(),
                              expected.c_str(), last_depth);
            }
            std::string out = line;
            if (e.ce) {
                const int diff = last_score - *e.ce;
                ce_abs_sum += std::abs(diff);
                ++ce_count;
                out += "  cp " + std::to_string(last_score) + " vs " + std::to_string(*e.ce);
            }
            send(out);
        }

        const int tested = static_cast<int>(suite.size()) - skipped;
        char pct[16];
        std::snprintf(pct, sizeof pct, "%.1f", tested ? 100.0 * solved / tested : 0.0);
        send("===========================");
        send("Solved          : ", solved, " / ", tested, " (", pct, "%)",
             skipped ? "  skipped " + std::to_string(skipped) : std::string());
        send("Avg time found  : ", solved ? found_time_sum / solved : 0, " ms  (limit ", ms, " ms)");
        send("Total nodes     : ", total_nodes);
        if (ce_count)
            send("Avg cp drift    : ", ce_abs_sum / ce_count, " (mean |ours - ce| over ", ce_count, ")");
        std::string list;
        for (const auto& id : failed) list += ' ' + id;
        send("Failed          :", list.empty() ? " none" : list);
    }

    // ---- search thread management ----------------------------------------
    void start_search(SearchLimits lim) {
        const bool white = position_.white_to_move();
        const TimeBudget budget = allocate_time(lim, white, options_.move_overhead);

        stop_ = false;
        pondering_ = lim.ponder;
        start_ms_ = now_ms();
        tt_.new_search();

        worker_ = std::thread([this, pos = position_, opts = options_,
                               lim = std::move(lim), budget]() mutable {
            SearchContext ctx{std::move(pos), std::move(lim), opts, budget,
                              stop_, pondering_, start_ms_, tt_,
                              search::KeepHistory ? tables_.get() : nullptr, &helper_tables_};
            SearchResult res = run_search(ctx);

            // UCI: in "infinite" or "ponder" mode, bestmove must not be sent
            // until the GUI says stop (or ponderhit for ponder).
            {
                std::unique_lock lock(wait_mutex_);
                wait_cv_.wait(lock, [&] {
                    return stop_.load() || !(ctx.limits.infinite || pondering_.load());
                });
            }

            if (res.ponder.empty()) send("bestmove ", res.best);
            else                    send("bestmove ", res.best, " ponder ", res.ponder);
        });
    }

    void stop_search() {
        {
            std::lock_guard lock(wait_mutex_);
            stop_ = true;
        }
        wait_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        stop_ = false;
    }

    PositionCmd position_;
    Options options_;
    search::TranspositionTable tt_{16};  // matches the Hash option default
    std::unique_ptr<search::OrderTables> tables_ = std::make_unique<search::OrderTables>();
    std::vector<std::unique_ptr<search::OrderTables>> helper_tables_;  // HelperTables: one per helper thread

    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pondering_{false};
    std::atomic<TimeMs> start_ms_{0};
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
};

}  // namespace uci

// Search.h : Search for GrandMassive.
//
// Iterative deepening -> Principal Variation Search (negamax alpha-beta)
// -> quiescence search at the leaves.
//
// Transposition table (TT.h): cutoffs in non-PV nodes.
// Move ordering: hash move (PV move at the root), then winning / equal
// captures by MVV-LVA, then killer moves, then quiet moves by history
// heuristic, then losing captures (by static exchange evaluation, SEE).
// Quiescence search: losing captures skipped, delta pruning.
// Selectivity: check extension, reverse futility pruning, null-move pruning,
// ProbCut, late move pruning, late move reductions. The last five use the
// "improving" flag (static eval better than on our previous move).

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Chess.h"
#include "Eval.h"
#include "TT.h"

namespace search {

using eval::Score;
using eval::Infinite;
using eval::Mate;

// ---- Tunable search parameters ------------------------------------------------
// SP_PARAM(name, value, min, max): a compile-time constant in normal builds.
// Built with -DSP_TUNE=1, each is a variable exposed as a UCI spin option of
// the same name, so an SPSA tuner can set it between games (tools/spsa.cpp).
//
// SPSA round 1: all 30 parameters, 6000 game pairs @20 ms/move (Fishtest-style
// schedules, c_end = range/20, r_end = 0.002). Largest moves: LMR base
// 0.75 -> 0.84, divisor 2.25 -> 2.04, first reduced move 3 -> 2; null move
// min depth 3 -> 2, base R 3 -> 4; RFP margin 80 -> 72; history pruning
// 2000 -> 1797; history-LMR divisor 4096 -> 4403. The values below are the
// tuned ones. A/B vs. the untuned engine: 800 games @25 ms +307 -218 =275
// (~+39), 600 games @100 ms +191 -137 =272 (~+31, LOS 99.8%).     ADOPTED
// SPSA round 2 (after the self-play eval re-tune), same setup, openings from
// tools/openings.h: LMR base 0.84 -> 0.90, divisor 2.04 -> 1.89; RFP margin
// 72 -> 68; LMP scale 2.02 -> 2.13; singular min depth 8 -> 7; IIR depth
// 6 -> 5; futility base 99 -> 95. A/B vs. round 1 values: 600 games @25 ms
// +206 -148 =246 (~+34), 600 @100 ms +176 -133 =291 (~+25, LOS 99%).  ADOPTED
// SPSA round 3 (on NNUE net 2), same setup: RFP margin 68 -> 63; null move
// min depth 2 -> 1; singular min depth 7 -> 6; IIR depth 5 -> 4; history
// pruning 1774 -> 1509; LMR divisor 1.89 -> 1.81; futility base 95 -> 90.
// A/B vs. round 2 values (both net 2): 600 games @25 ms +219 -159 =222
// (~+35), 600 @100 ms +158 -134 =308 (~+14, LOS 92%).                 ADOPTED
#ifndef SP_TUNE
#define SP_TUNE 0
#endif
struct TuneParam {
    const char* name;
    int* value;
    int def, min, max;
};
inline std::vector<TuneParam>& tune_params() {
    static std::vector<TuneParam> v;
    return v;
}
inline bool tune_register(const char* name, int* value, int def, int min, int max) {
    tune_params().push_back({name, value, def, min, max});
    return true;
}
#if SP_TUNE
#define SP_PARAM(name, value, lo, hi) \
    inline int name = (value);        \
    inline const bool name##_registered = tune_register(#name, &name, (value), (lo), (hi));
#else
#define SP_PARAM(name, value, lo, hi) inline constexpr int name = (value);
#endif

// Search statistics for the "orderstats" command (off otherwise): where in
// the move list the cutoff move was (Stockfish reports the same through its
// debug counters), and how often reductions / pruning had to be undone.
struct SearchStats {
    // Cutoffs at non-root nodes. idx: position in the move list (pruned moves
    // included). Categories by order band: 0 TT move, 1 good capture / queen
    // promotion, 2 killer 1, 3 killer 2, 4 counter, 5 quiet by history,
    // 6 bad capture, 7 under-promotion.
    long cutoffs = 0, first = 0, top3 = 0, idx_sum = 0, nonpv = 0, nonpv_first = 0;
    long cat[8] = {}, cat_idx[8] = {}, late = 0, late_cat[8] = {}, fail_first_cat[8] = {};
    long nodes_main = 0, nodes_qs = 0;
    long lmr = 0, lmr_research = 0;       // reduced searches, and those that beat alpha (re-searched)
    long nmp = 0, nmp_cut = 0;            // null-move searches, and cutoffs
    long rfp = 0, razor = 0;              // reverse futility / razoring cutoffs
    long pc = 0, pc_cut = 0;              // ProbCut nodes tried, cutoffs
    long lmp = 0, fut = 0, see_q = 0, hist = 0, see_c = 0;  // moves pruned by each rule
    long se = 0, se_ext = 0, se_multicut = 0;               // singular searches, extensions, multi-cuts
    long iid = 0, iid_move = 0;          // IID searches, and those that found a move
    long nnue_ord = 0, nnue_evals = 0;   // nodes NNUE-reordered, evals spent
    long tt_probes = 0, tt_found = 0, tt_hit = 0, tt_move = 0;  // main search: probes, key found (any), with a bound, with a move
};
inline SearchStats g_stats;
inline bool g_collect_stats = false;
#define SEARCH_STAT(x) \
    do {                \
        if (g_collect_stats) { x; } \
    } while (0)

inline constexpr int MaxPly = 128;

// Mate scores are Mate - ply (we mate) or -Mate + ply (we get mated).
inline bool is_mate_score(Score s) { return std::abs(s) >= Mate - MaxPly; }

// Moves until mate for UCI "score mate N" (positive = we mate, negative = we
// get mated). Only meaningful when is_mate_score(s).
inline int mate_in_moves(Score s) {
    return s > 0 ? (Mate - s + 1) / 2 : -(Mate + s) / 2;
}

// Mate scores depend on the distance from the root, but a TT entry can be
// reached at a different ply. Store them relative to the node instead.
inline Score score_to_tt(Score s, int ply) {
    if (s >= Mate - MaxPly) return s + ply;
    if (s <= -(Mate - MaxPly)) return s - ply;
    return s;
}
inline Score score_from_tt(Score s, int ply) {
    if (s >= Mate - MaxPly) return s - ply;
    if (s <= -(Mate - MaxPly)) return s + ply;
    return s;
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------
// Order scores, highest searched first. Bands keep each class of move apart:
inline constexpr int OrderHash       = 2'000'000;  // TT move / previous PV move
inline constexpr int OrderCapture    = 1'000'000;  // + MVV-LVA; also queen promotions
inline constexpr int OrderKiller1    =   900'000;
inline constexpr int OrderBadCapture = -500'000;  // + MVV-LVA; captures that lose material
inline constexpr int OrderKiller2    =   800'000;
inline constexpr int MaxHistory      =    16'384;  // quiets: history in [-Max, Max]
inline constexpr int OrderUnderPromo = -1'000'000;  // under-promotions: almost never best

inline bool is_quiet(const chess::Board& board, const chess::Move& m) {
    return m.typeOf() != chess::Move::PROMOTION && m.typeOf() != chess::Move::ENPASSANT &&
           !board.isCapture(m);
}

// MVV-LVA: prefer taking the most valuable victim, then with the least
// valuable attacker (PxQ before QxQ before QxP).
inline int mvv_lva(const chess::Board& board, const chess::Move& m) {
    const int victim = m.typeOf() == chess::Move::ENPASSANT
                           ? eval::PawnValue
                           : eval::PieceValueMG[static_cast<std::size_t>(board.at(m.to()).type().internal())];
    const int attacker = static_cast<int>(board.at(m.from()).type().internal());  // P=0 ... K=5
    return victim * 8 - attacker;
}

// ---------------------------------------------------------------------------
// Static exchange evaluation (SEE)
// ---------------------------------------------------------------------------
// see_ge(board, move, threshold): does `move` win at least `threshold`
// centipawns once all captures on its target square are played out, each
// side always recapturing with its least valuable piece and free to stop?
// Includes x-rays (a rook behind a rook joins in once the front one moves).
// Pins are ignored. Castling, en passant and promotions count as 0.
inline eval::Score see_value(chess::PieceType pt) {
    constexpr std::array<eval::Score, 6> v = {
        eval::PawnValue, eval::KnightValue, eval::BishopValue,
        eval::RookValue, eval::QueenValue, 20000,  // king: capturing it ends the exchange
    };
    return v[static_cast<std::size_t>(pt.internal())];
}

// All pieces of both colours attacking `sq`, given occupancy `occ`.
inline chess::Bitboard attackers_to(const chess::Board& b, chess::Square sq, chess::Bitboard occ) {
    using chess::Color;
    using chess::PieceType;
    using at = chess::attacks;
    const chess::Bitboard queens = b.pieces(PieceType::QUEEN);
    return (at::pawn(Color::BLACK, sq) & b.pieces(PieceType::PAWN, Color::WHITE)) |
           (at::pawn(Color::WHITE, sq) & b.pieces(PieceType::PAWN, Color::BLACK)) |
           (at::knight(sq) & b.pieces(PieceType::KNIGHT)) |
           (at::bishop(sq, occ) & (b.pieces(PieceType::BISHOP) | queens)) |
           (at::rook(sq, occ) & (b.pieces(PieceType::ROOK) | queens)) |
           (at::king(sq) & b.pieces(PieceType::KING));
}

inline bool see_ge(const chess::Board& b, const chess::Move& m, eval::Score threshold) {
    using chess::Bitboard;
    using chess::PieceType;

    if (m.typeOf() != chess::Move::NORMAL) return 0 >= threshold;

    const chess::Square from = m.from(), to = m.to();

    // After our capture we are up `swap` relative to the threshold.
    eval::Score swap = (b.at(to) == chess::Piece::NONE ? 0 : see_value(b.at(to).type())) - threshold;
    if (swap < 0) return false;  // even keeping the victim isn't enough

    // If they recapture our piece for free, are we still at or above it?
    swap = see_value(b.at(from).type()) - swap;
    if (swap <= 0) return true;

    Bitboard occ = b.occ() ^ Bitboard::fromSquare(from) ^ Bitboard::fromSquare(to);
    chess::Color stm = b.at(from).color();
    Bitboard attackers = attackers_to(b, to, occ);
    const Bitboard diag = b.pieces(PieceType::BISHOP) | b.pieces(PieceType::QUEEN);
    const Bitboard orth = b.pieces(PieceType::ROOK) | b.pieces(PieceType::QUEEN);
    int res = 1;

    for (;;) {
        stm = ~stm;
        attackers &= occ;
        const Bitboard mine = attackers & b.us(stm);
        if (mine.empty()) break;
        res ^= 1;

        // Least valuable attacker of the side to move.
        auto take = [&](PieceType pt) -> bool {
            const Bitboard bb = mine & b.pieces(pt);
            if (bb.empty()) return false;
            swap = see_value(pt) - swap;
            occ ^= Bitboard::fromSquare(bb.lsb());
            return true;
        };
        if (take(PieceType::PAWN)) {
            if (swap < res) break;
            attackers |= chess::attacks::bishop(to, occ) & diag;
        } else if (take(PieceType::KNIGHT)) {
            if (swap < res) break;
        } else if (take(PieceType::BISHOP)) {
            if (swap < res) break;
            attackers |= chess::attacks::bishop(to, occ) & diag;
        } else if (take(PieceType::ROOK)) {
            if (swap < res) break;
            attackers |= chess::attacks::rook(to, occ) & orth;
        } else if (take(PieceType::QUEEN)) {
            if (swap < res) break;
            attackers |= (chess::attacks::bishop(to, occ) & diag) | (chess::attacks::rook(to, occ) & orth);
        } else {
            // King: it may only capture if the other side has no attackers left.
            return (attackers & ~b.us(stm)).empty() ? res : res ^ 1;
        }
    }
    return res != 0;
}

// Is `m` (e.g. a TT move, which may come from a key collision) legal here?
// Generates only the legal moves of the piece type on its from-square.
inline bool is_legal_move(const chess::Board& b, const chess::Move& m) {
    if (m == chess::Move::NO_MOVE) return false;
    const chess::Piece pc = b.at(m.from());
    if (pc == chess::Piece::NONE || pc.color() != b.sideToMove()) return false;
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, b, 1 << static_cast<int>(pc.type().internal()));
    for (const auto& x : ml)
        if (x == m) return true;
    return false;
}

// Does `m` give check? Direct attack from the destination, or a discovered
// attack by one of our sliders. Castling and en passant: make the move.
inline bool gives_check(const chess::Board& b, const chess::Move& m) {
    using chess::PieceType;
    if (m.typeOf() == chess::Move::CASTLING || m.typeOf() == chess::Move::ENPASSANT) {
        chess::Board c = b;
        c.makeMove(m);
        return c.inCheck();
    }
    const chess::Color us = b.sideToMove();
    const chess::Square ksq = b.kingSq(~us);
    const std::uint64_t k = 1ULL << ksq.index();
    const std::uint64_t from = 1ULL << m.from().index(), to = 1ULL << m.to().index();
    const chess::Bitboard occ((b.occ().getBits() & ~from) | to);
    const PieceType pt = m.typeOf() == chess::Move::PROMOTION ? m.promotionType() : b.at(m.from()).type();
    std::uint64_t direct = 0;
    if (pt == PieceType::PAWN) direct = chess::attacks::pawn(us, m.to()).getBits();
    else if (pt == PieceType::KNIGHT) direct = chess::attacks::knight(m.to()).getBits();
    else if (pt == PieceType::BISHOP) direct = chess::attacks::bishop(m.to(), occ).getBits();
    else if (pt == PieceType::ROOK) direct = chess::attacks::rook(m.to(), occ).getBits();
    else if (pt == PieceType::QUEEN) direct = chess::attacks::queen(m.to(), occ).getBits();
    if (direct & k) return true;
    const std::uint64_t diag = (b.pieces(PieceType::BISHOP, us) | b.pieces(PieceType::QUEEN, us)).getBits() & ~from;
    const std::uint64_t orth = (b.pieces(PieceType::ROOK, us) | b.pieces(PieceType::QUEEN, us)).getBits() & ~from;
    return (chess::attacks::bishop(ksq, occ).getBits() & diag) || (chess::attacks::rook(ksq, occ).getBits() & orth);
}

// Hash of a bitboard set, for the correction-history tables.
inline std::uint64_t mix_key(std::uint64_t a, std::uint64_t b, std::uint64_t c = 0) {
    std::uint64_t h = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL ^
                      (c + 0x85EBCA77C2B2AE63ULL) * 0x165667B19E3779F9ULL;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    return h ^ (h >> 29);
}

// History heuristic: quiet moves that caused beta cutoffs, by side/from/to.
// "Gravity" update keeps values within [-MaxHistory, MaxHistory] and lets
// old information fade as new results come in.
class History {
public:
    int get(chess::Color c, const chess::Move& m) const {
        return table_[static_cast<std::size_t>(c.internal())][sq(m.from())][sq(m.to())];
    }
    void update(chess::Color c, const chess::Move& m, int bonus) {
        int& h = table_[static_cast<std::size_t>(c.internal())][sq(m.from())][sq(m.to())];
        bonus = std::clamp(bonus, -MaxHistory, MaxHistory);
        h += bonus - h * std::abs(bonus) / MaxHistory;
    }

private:
    static std::size_t sq(chess::Square s) { return static_cast<std::size_t>(s.index()); }
    std::array<std::array<std::array<int, 64>, 64>, 2> table_{};
};

// Same gravity update for the 16-bit tables below.
inline void update_stat(std::int16_t& h, int bonus) {
    bonus = std::clamp(bonus, -MaxHistory, MaxHistory);
    const int v = h;
    h = static_cast<std::int16_t>(v + bonus - v * std::abs(bonus) / MaxHistory);
}

// (piece, to-square) key for a move: 0..767, or -1 for "no move" (null move,
// root). Castling uses the king's from/to encoding of the library, which
// is consistent, so it works as a key too.
inline int piece_to(const chess::Board& b, const chess::Move& m) {
    return static_cast<int>(b.at(m.from()).internal()) * 64 + m.to().index();
}

// Move ordering, round 2 (Oct 2026). Stockfish 19's quiet cutoffs come at
// mean position 1.64-1.69 (instrumented, depth 13 / 15), ours (killers,
// counter and history quiets together) at 2.35-2.37.
// ContSplit: one continuation-history table per ply offset (1, 2, 3, 4, 6
// plies back) instead of one shared table. Offsets 1 and 2 never collided in
// the shared table (the previous move's piece has the other colour 1 ply back
// and ours 2 plies back), so with the defaults nothing changes (identical
// bench); only 1/3 and 2/4/6 shared entries. With ContOrdW3/4/6 = 16/32/16 on
// top: 8.06M nodes, quiet history at mean position 4.36: no better.
SP_PARAM(ContSplit, 0, 0, 1)
// ThreatHist: the main (butterfly) history also indexed by whether the move's
// from- and to-square are attacked by the opponent: the same move is good
// when it escapes an attack and bad when it walks into one. Orderstats depth
// 13 (base 7.11M nodes, mean cutoff position 1.521, quiet history 4.29):
// 6.81M nodes (-4.2%), 1.508, quiet history 4.09, LMR re-search 1.35% ->
// 1.26%. With ContSplit + continuation 3/4/6 as well: 7.70M, quiets 3.99.
SP_PARAM(ThreatHist, 0, 0, 1)
// TtBuckets: transposition table in buckets of 4 slots per cache line,
// replacing the least valuable entry, instead of one slot per key that keeps
// the deeper entry (TT.h). Set on the table by every Searcher.
// Orderstats depth 13: at the bench's 16 MB there is no pressure (one slot:
// 67% of stores into empty slots, 3.6% dropped; found 50.6% either way), so
// the found-rate gap to Stockfish is not replacement. At 1 MB: one slot drops
// 28.7% of stores, found 38.7%; buckets drop 0.6%, found 44.3%.
// A/B (Linux, 2 games at a time), 2+0.02 with Hash 2 on both (pressure like
// a long time control): +4.5 +- 11.3 after 1160 games, LLR +0.30; stopped.
// SPRT on the PC, Hash 4 on both, [0, 5]: H1, +6.2 Elo after 4200 games.
// ADOPTED (Oct 2026). Bench 262010 -> 262029.
SP_PARAM(TtBuckets, 1, 0, 1)
// RepRule: 0 = any earlier occurrence of the position, game history before
// the search included, is a draw (chess::Board::isRepetition(1)); 1 = only an
// occurrence inside the search tree, or a third occurrence (as Stockfish).
// With 0 a position that occurred once earlier in the game scores 0 in the
// search though the opponent can simply play on. Oct 2026, 88 games vs
// Stockfish 8 at 2+1: in several lost games our eval sat at exactly 0.00 for
// 10+ moves while Stockfish showed -1.1 for us, then the position collapsed.
// Self-play SPRT [0, 5] at 10+0.1: about flat (both sides share the rule;
// it only matters when a line passes a position of the game), stopped.
// ADOPTED on correctness (Stockfish's rule). Bench unchanged (no game history).
SP_PARAM(RepRule, 1, 0, 1)
// FailLowMove: when every move fails low the node has no best move, so its
// TT entry carries none (Oct 2026: only ~19% of main-search probes returned a
// move). 1 = store the best-scoring move anyway (an upper bound, but the
// likeliest move to try first next time); 2 = only when the node had no TT
// move to begin with (never overwrite a move from an earlier search).
// Orderstats depth 13 (bench positions; base 5.64M nodes, first-move cutoffs
// 85.0%, TT probes with a move 17.3%): FailLowMove 1: 7.72M nodes, 85.6%,
// 22.2%; 2: 7.20M, 85.6%, 22.6%. The extra nodes come from IIR: it reduces
// only nodes without a TT move, and now fewer have none (see FailLowIir).
// SPRT 10+0.1 (Oct 9 2026): mode 2 about 0 Elo after many games, no H1. Off.
SP_PARAM(FailLowMove, 0, 0, 2)
// FailLowIir: IIR still reduces when the TT move came from a fail-low
// (upper-bound) entry, so FailLowMove changes move ordering only, not depth.
// Alone it also reduces nodes whose fail-low entry kept an older move:
// orderstats depth 13 3.45M nodes (-39%), depth 15 7.89M vs 13.58M (-42%),
// first-move cutoffs 85.0% -> 85.2%. With FailLowMove 2: 3.60M / 8.14M.
// SPRT 10+0.1 (Oct 9 2026): about -40 Elo after 120+ games (over-reduction:
// most non-PV entries are upper bounds). Stopped. Off.
SP_PARAM(FailLowIir, 0, 0, 1)
// PriorCmBonus: when a node fails low, the opponent's quiet move that led to
// it worked (it cut off one ply up), so that move gets an extra history bonus
// (main + continuation history) of PriorCmBonus % of history_bonus(depth).
// Orderstats depth 13: 100: 7.16M nodes (+27%), quiet cutoff mean position
// 4.55 -> 4.23, LMR re-searches 1.18% -> 1.10%; 200: 7.74M, 4.20, 0.99%.
// The extra nodes: higher history values mean smaller LMR reductions
// (HistLmrDivisor was tuned without it).
SP_PARAM(PriorCmBonus, 0, 0, 400)
// LowPlyHist: an extra quiet-move history for the first LowPlyN plies of the
// search, fresh every search, added to move ordering there with weight
// LowPlyHist/64. Near the root the same positions are searched every
// iteration, so their own statistics beat the global tables.
// Orderstats depth 13: 64: 5.95M nodes (+5.5%), ordering about unchanged;
// 128: 5.95M, first-move cutoffs 85.0% -> 85.4%.
SP_PARAM(LowPlyHist, 0, 0, 256)
SP_PARAM(LowPlyN, 4, 1, 8)

// All move-ordering statistics. Owned by the UCI layer so they survive from
// one move of the game to the next (cleared on ucinewgame); a Searcher made
// without one creates its own.
struct OrderTables {
    History main;  // [color][from][to]
    // Capture history: [moving piece][to][captured type]. Refines MVV-LVA by
    // which captures actually worked in this game.
    std::array<std::array<std::array<std::int16_t, 6>, 64>, 12> capture{};
    // Continuation history: [previous move's piece-to][this move's piece-to].
    // "After the opponent played Nf6, h3 tends to be good." Used with the
    // moves 1 ply ago (counter-move history) and 2 plies ago (follow-up).
    std::vector<std::int16_t> cont = std::vector<std::int16_t>(768 * 768);
    // Countermove: the quiet move that last refuted the opponent's piece-to.
    std::array<chess::Move, 768> counter{};

    // Correction history (SR_CORR): how far the static eval tends to be off,
    // by pawn structure and by each side's non-pawn pieces, per side to move.
    static constexpr std::size_t CorrSize = 16384;
    std::vector<std::int16_t> corr_pawn = std::vector<std::int16_t>(2 * CorrSize);
    std::vector<std::int16_t> corr_np = std::vector<std::int16_t>(2 * 2 * CorrSize);
    // Audit additions: correction by minor pieces (+ kings) per side to move,
    // and by the previous move's piece-to (continuation correction).
    std::vector<std::int16_t> corr_minor = std::vector<std::int16_t>(2 * CorrSize);
    std::vector<std::int16_t> corr_cont = std::vector<std::int16_t>(2 * 768);
    // Pawn-structure history: quiet moves (piece-to) by pawn structure.
    static constexpr std::size_t PawnHistSize = 512;
    std::vector<std::int16_t> pawn_hist = std::vector<std::int16_t>(PawnHistSize * 768);

    // ContSplit: one table per offset (1, 2, 3, 4, 6 plies back -> 0..4).
    std::vector<std::int16_t> cont_split = std::vector<std::int16_t>(5 * 768 * 768);
    // ThreatHist: [color][from attacked][to attacked][from][to].
    std::vector<std::int16_t> threat_hist = std::vector<std::int16_t>(2 * 4 * 64 * 64);
    static int offset_index(int back) { return back == 6 ? 4 : back - 1; }
    std::int16_t& cont_at(int prev, int cur, int back) {
        return ContSplit ? cont_split[static_cast<std::size_t>((offset_index(back) * 768 + prev) * 768 + cur)]
                         : cont[static_cast<std::size_t>(prev * 768 + cur)];
    }
    int cont_get(int prev, int cur, int back) const {
        if (prev < 0) return 0;
        return ContSplit ? cont_split[static_cast<std::size_t>((offset_index(back) * 768 + prev) * 768 + cur)]
                         : cont[static_cast<std::size_t>(prev * 768 + cur)];
    }
    std::int16_t& cont_at(int prev, int cur) { return cont[static_cast<std::size_t>(prev * 768 + cur)]; }
    int cont_get(int prev, int cur) const {
        return prev < 0 ? 0 : cont[static_cast<std::size_t>(prev * 768 + cur)];
    }
    std::int16_t& capture_at(const chess::Board& b, const chess::Move& m) {
        const int victim = m.typeOf() == chess::Move::ENPASSANT ? 0 : static_cast<int>(b.at(m.to()).type().internal());
        return capture[static_cast<std::size_t>(b.at(m.from()).internal())][static_cast<std::size_t>(m.to().index())]
                      [static_cast<std::size_t>(victim)];
    }
    int capture_get(const chess::Board& b, const chess::Move& m) const {
        return const_cast<OrderTables*>(this)->capture_at(b, m);
    }
};

// Moves plus their order scores. next() does one step of selection sort:
// only as much sorting as the search actually uses before a cutoff.
#ifndef SP_SORT_AFTER
#define SP_SORT_AFTER 0     // 0 = off; n = sort the rest of the list once n moves were picked
#endif
inline constexpr bool UseSortAfter = SP_SORT_AFTER > 0;
inline constexpr int SortAfter = SP_SORT_AFTER;

struct OrderedMoves {
    chess::Movelist moves;
    std::array<int, 256> scores;  // filled by score_moves() before next() is used

    int size() const { return static_cast<int>(moves.size()); }

    // Picks the best remaining move into slot i (selection sort: most nodes
    // cut off after one or two moves). SP_SORT_AFTER: once that many moves
    // were picked (probably an all-node, which tries everything), sort the
    // rest once (stable insertion sort) instead of scanning for each pick.
    chess::Move next(int i) {
        if (UseSortAfter && i >= SortAfter) {
            if (i == SortAfter) {
                for (int a = i + 1; a < size(); ++a) {
                    const chess::Move m = moves[a];
                    const int sc = scores[static_cast<std::size_t>(a)];
                    int b = a - 1;
                    for (; b >= i && scores[static_cast<std::size_t>(b)] < sc; --b) {
                        moves[b + 1] = moves[b];
                        scores[static_cast<std::size_t>(b + 1)] = scores[static_cast<std::size_t>(b)];
                    }
                    moves[b + 1] = m;
                    scores[static_cast<std::size_t>(b + 1)] = sc;
                }
            }
            return moves[i];
        }
        int best = i;
        for (int j = i + 1; j < size(); ++j)
            if (scores[static_cast<std::size_t>(j)] > scores[static_cast<std::size_t>(best)]) best = j;
        std::swap(moves[i], moves[best]);
        std::swap(scores[static_cast<std::size_t>(i)], scores[static_cast<std::size_t>(best)]);
        return moves[i];
    }
};

// ---------------------------------------------------------------------------
// Pruning / reduction parameters (untuned starting points, in centipawns
// where applicable). Tune these with SPRT games, not by eye.
// ---------------------------------------------------------------------------

// Reverse futility pruning: at depth <= RfpMaxDepth, a non-PV node whose
// static eval beats beta by RfpMargin per ply of depth is cut off.
SP_PARAM(RfpMaxDepth, 8, 4, 12)
SP_PARAM(RfpMargin, 68, 30, 160)

// Null-move pruning: from this depth, reduction R = NmpBase + depth / NmpDiv.
SP_PARAM(NmpMinDepth, 1, 1, 6)
SP_PARAM(NmpBase, 4, 1, 6)
SP_PARAM(NmpDiv, 6, 3, 12)

// ---- "Improving" and the heuristics that use it ----------------------------
// improving = our static eval is higher than it was two plies ago (our
// previous move). If improving, the position is trending our way, so we can
// prune more aggressively toward fail-highs and reduce less.
// The SR_* macros exist so A/B test builds can flip features from the
// command line; normal builds use the defaults.
//
// Timed A/B results (25 ms/move with 5 ms overhead, vs. the version before):
//   LMP, Stockfish limit (3+d^2)/(2-imp)     -250-ish (aborted), with checks kept: similar
//   LMP, same limit, depth <= 3 only           -32  (400 games)
//   LMP, limit x2, checks kept                 +23  (400, LOS 95%)   ON
//   then, each on top of that LMP:
//   RFP margin by improving                     -2  (600)            off
//     re-tested @100 ms after the speed round + SPSA: +188 -151 =261
//     (600, ~+21, LOS ~98%)                                           ON
//   LMR +1 when not improving                   +3  (600)            off
//   NMP below beta when improving               -1  (600)            off
//   ProbCut                                     -4  (600)            off
//   all four together                           +7  (800, LOS 77%)   off: not proven
// The four are standard small gains in strong engines; they need an SPRT
// run of thousands of games to resolve. Aggressive LMP likely fails here
// because quiet-move ordering is still basic (no continuation history).
#ifndef SR_IMP_RFP
#define SR_IMP_RFP 1
#endif
#ifndef SR_LMP
#define SR_LMP 1
#endif
#ifndef SR_IMP_LMR
#define SR_IMP_LMR 0
#endif
#ifndef SR_PROBCUT
#define SR_PROBCUT 0
#endif
#ifndef SR_LMP_KEEP_CHECKS
#define SR_LMP_KEEP_CHECKS 1
#endif
inline constexpr bool UseImprovingRfp = SR_IMP_RFP;  // RFP margin: RfpMargin * (depth - improving)
inline constexpr bool UseLmp          = SR_LMP;      // late move pruning, limit depends on improving
inline constexpr bool UseImprovingLmr = SR_IMP_LMR;  // +1 reduction when not improving
inline constexpr bool UseProbCut      = SR_PROBCUT;  // ProbCut, beta margin lower when improving
inline constexpr bool LmpKeepChecks   = SR_LMP_KEEP_CHECKS;  // LMP never skips a move that gives check

// Late move pruning: at depth <= LmpMaxDepth, quiet moves after the first
// (3 + depth^2) / (2 - improving) are skipped.
#ifndef SR_LMP_MAX_DEPTH
#define SR_LMP_MAX_DEPTH 8
#endif
#ifndef SR_LMP_SCALE
#define SR_LMP_SCALE 2
#endif
SP_PARAM(LmpMaxDepth, SR_LMP_MAX_DEPTH, 3, 12)
SP_PARAM(LmpScale, 215, 80, 400)  // x100
inline int lmp_limit(int depth, bool improving) {
    return LmpScale * (3 + depth * depth) / (100 * (2 - (improving ? 1 : 0)));
}

// ---- Move ordering statistics ------------------------------------------------
// Timed A/B results (25 ms/move, 600 games each):
//   vs. the version before (fresh, small-bonus history every move):
//     capture history                           -7            off
//     continuation history (1 and 2 ply)        +5
//     countermove                               +9
//     history bonus 16d^2+32d instead of d^2    +6
//     keep tables between moves                +28 (LOS 99%)  ON
//   then vs. keep-tables:
//     continuation history + countermove        -7  (with the small d^2 bonus)
//     big bonus                                +21 (LOS 97%)  ON
//     big bonus + cont + counter + history LMR +37 (LOS 99.9%) ON: the package
//     package vs. big bonus alone              +13 (LOS 87%)
//     package + capture history                 -7            off
// Continuation history and history-LMR need the larger bonus: with d^2 the
// values stay too small to matter. Total over the old ordering: about +65.
#ifndef SR_CAPT_HIST
#define SR_CAPT_HIST 0
#endif
#ifndef SR_CONT_HIST
#define SR_CONT_HIST 1
#endif
#ifndef SR_COUNTER
#define SR_COUNTER 1
#endif
#ifndef SR_HIST_LMR
#define SR_HIST_LMR 1
#endif
#ifndef SR_BIG_BONUS
#define SR_BIG_BONUS 1
#endif
// ---- Extensions ------------------------------------------------------------
// A/B (vs. the engine without them, 25 ms/move unless noted):
//   singular extension     openings -2 (600), EG suite -2 (600);
//                          at 100 ms/move -12 (450: +133 =169 -148)          off
//   passed-pawn extension  openings +6 (600, LOS 71%), EG suite -5 (600)    off
// Neither is a proven gain here; kept, switched off, for retesting at longer
// time controls or after other search changes.
#ifndef SR_SINGULAR
#define SR_SINGULAR 1
#endif
#ifndef SR_PAWN_EXT
#define SR_PAWN_EXT 0
#endif
// Singular extension: at depth >= SingularMinDepth, if the TT move (a lower
// bound or exact, from a search at most 3 plies shallower) is better than
// every other move by SingularMargin * depth in a half-depth search, it is
// searched one ply deeper. If even the other moves beat beta, cut off
// ("multi-cut").
inline constexpr bool UseSingular = SR_SINGULAR;
SP_PARAM(SingularMinDepth, 6, 5, 12)
SP_PARAM(SingularMarginX8, 14, 4, 48)  // singular beta = TT score - margin/8 * depth
SP_PARAM(DoubleExtMargin, 22, 0, 80)   // double extension when this far below singular beta
// Passed-pawn extension: a pawn push to the 7th rank is searched one ply deeper.
inline constexpr bool UsePawnExt = SR_PAWN_EXT;

#ifndef SR_KEEP_HIST
#define SR_KEEP_HIST 1
#endif
inline constexpr bool UseCaptureHistory = SR_CAPT_HIST;  // capture history on top of MVV
inline constexpr bool UseContHistory    = SR_CONT_HIST;  // 1- and 2-ply continuation history
inline constexpr bool UseCountermove    = SR_COUNTER;    // countermove after the killers
inline constexpr bool UseHistoryLmr     = SR_HIST_LMR;   // reduce good-history quiets less
inline constexpr bool KeepHistory       = SR_KEEP_HIST;  // tables survive between moves
inline constexpr bool UseBigBonus       = SR_BIG_BONUS;  // history bonus 16d^2+32d (cap 1600) instead of d^2
inline constexpr int OrderCounter    = 700'000;

// History bonus for a cutoff at `depth` (penalty = the negative).
SP_PARAM(HistBonusMul, 19, 4, 48)
SP_PARAM(HistBonusAdd, 41, 0, 128)
SP_PARAM(HistBonusMax, 1486, 400, 4000)
inline int history_bonus(int depth) {
    return UseBigBonus ? std::min(HistBonusMul * depth * depth + HistBonusAdd * depth, HistBonusMax) : depth * depth;
}
#ifndef SR_HIST_LMR_DIV
#define SR_HIST_LMR_DIV 6478
#endif
SP_PARAM(HistLmrDivisor, SR_HIST_LMR_DIV, 1024, 16384)  // history units per ply of LMR

inline constexpr int NmpImprovingMargin = 40;

// ---- Search round 2 (ideas from Stockfish 19's search; own implementation
// and values). Each is a switch, tested against the engine before it in three
// levels: L1 300 games @10 ms/move (variants screened), L2 600 @10 ms,
// L3 600 @25 ms (depth-gated features: all levels @25 ms). A feature that
// passes all three becomes the default and the base for the next one.
//   aspiration windows, 12 cp: L1 +84 (8 cp +49, 20 +49, 35 +38), L2 +67,
//     L3 +56                                                         ADOPTED
//   move-loop pruning (futility + SEE + history + razoring): L1 +89
//     (alone: futility +8, SEE +34, history -48, razoring +27), L2 +84,
//     L3 +70                                                         ADOPTED
//   cut-node group: LMR +1 at cut nodes -1, null move only at cut nodes +23
//     (L2 -5), null-move R 4+d/3+eval -2, IIR (25 ms) +15 / L2 +14 / L3 -6  off
//   correction history: L1 +5, L2 +9, L3 -3                             off
//   richer LMR (bad captures, killers, deeper/shallower re-search): L1 -23 off
//   ordering (check bonus, threat escape, 4-ply cont. history): -14; each
//     alone -33 / -22 / -20                                             off
//   mate distance pruning: L1 +14, L2 +10, L3 -10; fail-high blend -26;
//     no check extension -12                                            off
//   singular + negative + double extensions (25 ms): L1 +31, L2 -1, L3 -1 off
// After the first two, the rest were within about +-15 Elo, below what
// 600-game tests resolve; they stay here as switches for longer tests.
// Re-tested at 100 ms/move (800 games each vs. the engine with aspiration +
// pruning + time management):
//   singular + negative + double extensions ~+20 (LOS ~98%), IIR ~+15
//   (LOS ~94%), correction history ~+36 (LOS 99.9%); all three together
//   +305 -139 =356 (~+73, LOS 100%). Against Stockfish 19 (4000 nodes),
//   300 games, same openings: 23.0% vs. 19.5% (~+35).       ADOPTED (all three)
// They need depth to pay off (at 10-25 ms they were about even), and real
// games (e.g. 60+1) search much deeper than 100 ms.
#ifndef SR_ASP
#define SR_ASP 1            // aspiration windows at the root
#endif
#ifndef SR_ASP_DELTA
#define SR_ASP_DELTA 13     // initial half-width (cp)
#endif
#ifndef SR_IIR
#define SR_IIR 1            // internal iterative reduction: PV / cut node without a TT move
#endif
#ifndef SR_IIR_DEPTH
#define SR_IIR_DEPTH 4
#endif
#ifndef SR_LMR_CUT
#define SR_LMR_CUT 0        // +1 LMR at expected cut nodes
#endif
#ifndef SR_NMP_CUT
#define SR_NMP_CUT 0        // null move only at expected cut nodes
#endif
#ifndef SR_NMP2
#define SR_NMP2 0           // null-move R = 4 + depth/3 + min((eval-beta)/SR_NMP_EVDIV, 3)
#endif
#ifndef SR_NMP_EVDIV
#define SR_NMP_EVDIV 200
#endif
#ifndef SR_FUTP
#define SR_FUTP 1           // futility pruning of quiet moves (by reduced depth)
#endif
#ifndef SR_FUT_BASE
#define SR_FUT_BASE 90
#endif
#ifndef SR_FUT_MUL
#define SR_FUT_MUL 96
#endif
#ifndef SR_FUT_DEPTH
#define SR_FUT_DEPTH 8
#endif
#ifndef SR_SEEP
#define SR_SEEP 1           // SEE pruning: quiets below -SR_SEE_QUIET*d^2, captures below -SR_SEE_CAP*d
#endif
#ifndef SR_SEE_QUIET
#define SR_SEE_QUIET 23
#endif
#ifndef SR_SEE_CAP
#define SR_SEE_CAP 106
#endif
#ifndef SR_HISTP
#define SR_HISTP 1          // history pruning: quiets with history < -SR_HISTP_MUL*depth
#endif
#ifndef SR_HISTP_MUL
#define SR_HISTP_MUL 1509
#endif
#ifndef SR_HISTP_DEPTH
#define SR_HISTP_DEPTH 4
#endif
#ifndef SR_RAZOR
#define SR_RAZOR 1          // razoring: depth <= 3, eval + margin*depth < alpha -> qsearch
#endif
#ifndef SR_RAZOR_MARGIN
#define SR_RAZOR_MARGIN 244
#endif
#ifndef SR_CORR
#define SR_CORR 1           // correction history (pawn + non-pawn structure)
#endif
#ifndef SR_CORR_WP
#define SR_CORR_WP 58
#endif
#ifndef SR_CORR_WNP
#define SR_CORR_WNP 57
#endif
#ifndef SR_LMR2
#define SR_LMR2 0           // LMR: bad captures too, killers/counter less, deeper/shallower re-search
#endif
#ifndef SR_SE_NEG
#define SR_SE_NEG 1         // singular: negative extension when the TT move isn't singular
#endif
#ifndef SR_SE_DOUBLE
#define SR_SE_DOUBLE 1      // singular: double extension when far below singular beta
#endif
#ifndef SR_ORD2
#define SR_ORD2 0           // quiet ordering: check bonus, threat escape, 4-ply continuation history
#endif
#ifndef SR_ORD2_PARTS
#define SR_ORD2_PARTS 7     // bit 1 check bonus, 2 threat escape, 4 4-ply continuation history
#endif
#ifndef SR_MDP
#define SR_MDP 0            // mate distance pruning
#endif
#ifndef SR_FH_BLEND
#define SR_FH_BLEND 0       // blend a fail-high score towards beta
#endif
#ifndef SR_CHECK_EXT
#define SR_CHECK_EXT 1      // +1 ply in check
#endif
inline constexpr bool UseAsp = SR_ASP;
SP_PARAM(AspDelta, SR_ASP_DELTA, 5, 40)
inline constexpr bool UseIir = SR_IIR;
SP_PARAM(IirDepth, SR_IIR_DEPTH, 3, 10)
inline constexpr bool UseLmrCut = SR_LMR_CUT;
inline constexpr bool UseNmpCut = SR_NMP_CUT;
inline constexpr bool UseNmp2 = SR_NMP2;
inline constexpr int NmpEvalDiv = SR_NMP_EVDIV;
inline constexpr bool UseFutPrune = SR_FUTP;
SP_PARAM(FutBase, SR_FUT_BASE, 20, 250)
SP_PARAM(FutMul, SR_FUT_MUL, 40, 250)
SP_PARAM(FutDepth, SR_FUT_DEPTH, 3, 12)
inline constexpr bool UseSeePrune = SR_SEEP;
SP_PARAM(SeeQuietMul, SR_SEE_QUIET, 5, 60)
SP_PARAM(SeeCapMul, SR_SEE_CAP, 30, 250)
inline constexpr bool UseHistPrune = SR_HISTP;
SP_PARAM(HistPruneMul, SR_HISTP_MUL, 500, 8000)
SP_PARAM(HistPruneDepth, SR_HISTP_DEPTH, 2, 8)
inline constexpr bool UseRazor = SR_RAZOR;
SP_PARAM(RazorMargin, SR_RAZOR_MARGIN, 80, 600)
inline constexpr bool UseCorr = SR_CORR;
SP_PARAM(CorrWPawn, SR_CORR_WP, 8, 192)
SP_PARAM(CorrWNonPawn, SR_CORR_WNP, 8, 192)
inline constexpr bool UseLmr2 = SR_LMR2;
inline constexpr bool UseSeNeg = SR_SE_NEG;
inline constexpr bool UseSeDouble = SR_SE_DOUBLE;
inline constexpr bool UseOrd2 = SR_ORD2;
inline constexpr bool Ord2Check = UseOrd2 && (SR_ORD2_PARTS & 1);
inline constexpr bool Ord2Threat = UseOrd2 && (SR_ORD2_PARTS & 2);
inline constexpr bool Ord2Cont4 = UseOrd2 && (SR_ORD2_PARTS & 4);
inline constexpr bool UseMdp = SR_MDP;
inline constexpr bool UseFhBlend = SR_FH_BLEND;
inline constexpr bool UseCheckExt = SR_CHECK_EXT;
inline constexpr bool UseMovePruning = UseFutPrune || UseSeePrune || UseHistPrune;
inline constexpr int CorrLimit = 1024;  // correction-history entries in [-CorrLimit, CorrLimit]

// ---- Feature strengths (SPSA round 4) -------------------------------------------
// Switches that lost their A/B tests as on/off features, re-expressed as
// continuous strengths so SPSA can tune them from 0 (= off, the behaviour of
// the switch off). Each default reproduces its old SR_* switch (100 or the old
// value when the switch is on).
// SPSA round 4 (net 3, 43 parameters, these 13 started at 0) was stopped at
// 6280 of 12000 pairs: no parameter moved more than one perturbation step,
// the strengths stayed within 0-0.5 steps of 0 (ProbCut, capture history,
// check bonus, 4-ply continuation essentially 0) and the +/- pairs averaged
// -0.006. No signal from 0, consistent with the on/off A/B tests: all stay
// at 0 (disabled), kept in the code as tunable SP_PARAMs for later tests.
// SPSA round 5 (net 6): only these 13 tuned, everything else fixed, started at
// their "on" values, 5000 pairs @25 ms with r_end 0.02 (the default 0.002
// can't move a parameter one step in a run this short, which also explains
// round 4). Nothing went back to 0 (final: LmrCut 134, LmrNotImp 81,
// LmrRefuter 123, LmrBadCap 86, NmpAllMargin 38, NmpEvalScale 69, NmpImpMargin
// 34, ProbCut 105, CaptHistW 13, OrdCheckBonus 7252, OrdThreatX 12, Cont4W 26,
// FhBlend 75), but that set vs. all 13 off: +148 -193 =259 (600 games @25 ms,
// ~-26, LOS 1%). SPSA only finds the best point near where it starts; on/off
// is the A/B's call. All stay at 0.
// Move-ordering check (Oct 2026, bench 13; Stockfish dev instrumented the same
// way for comparison): share of cutoffs on the first move 86.7% (Stockfish
// 88.9% at depth 13, 87.9% at depth 15 with similar nodes), within 3 moves
// 96.1% (97.4 / 97.1), mean position of the cutoff move 1.47 (1.25 / 1.28).
// Weak spot: quiets ordered by history, 9% of cutoffs but 47% of the late
// ones, at mean position 4.1 (TT move, captures, killers: 1.0-1.2).
// OrdCheckBonus 16000 + OrdThreatX 16 bring that to 2.85 (overall 1.38), but
// still -10 in games (600 @25 ms, LOS 17%; 4.6% more nodes at fixed depth).
// Continuation 3/4/6, pawn history and capture history x128 don't improve
// the ordering numbers at all.
// Removed after the search audit (Oct 2026): NmpImpMargin (null move also
// when improving and eval >= beta - x) and NmpAllMargin (eval >= beta + x at
// non-cut nodes): not standard techniques, lost every test.
SP_PARAM(LmrCutX100, 85, 0, 200)      // LMR +x/100 ply at expected cut nodes
SP_PARAM(LmrNotImpX100, SR_IMP_LMR ? 100 : 0, 0, 200)   // LMR +x/100 ply when not improving
SP_PARAM(LmrRefuterX100, SR_LMR2 ? 100 : 0, 0, 200)     // LMR -x/100 ply for killers
SP_PARAM(LmrBadCapX100, SR_LMR2 ? 100 : 0, 0, 150)      // bad captures reduced by x% of the quiet reduction
SP_PARAM(NmpEvalScale, SR_NMP2 ? 100 : 0, 0, 200)       // null move R += min((eval - beta) * x / 20000, 3)
SP_PARAM(ProbCutStr, SR_PROBCUT ? 100 : 0, 0, 120)      // ProbCut on if > 0, margin 1000 - 8x (x = 100: 200)
SP_PARAM(CaptHistW, SR_CAPT_HIST ? 16 : 0, 0, 64)       // capture order += capture history * x / 256
SP_PARAM(OrdCheckBonus, (SR_ORD2 && (SR_ORD2_PARTS & 1)) ? 8000 : 0, 0, 16000)  // quiet checks (SEE >= -75)
SP_PARAM(OrdThreatX, (SR_ORD2 && (SR_ORD2_PARTS & 2)) ? 16 : 0, 0, 40)  // escaping / entering lesser-piece attacks
SP_PARAM(Cont4W, (SR_ORD2 && (SR_ORD2_PARTS & 4)) ? 32 : 0, 0, 64)     // 4-ply continuation history weight / 64
SP_PARAM(FhBlendX, SR_FH_BLEND ? 100 : 0, 0, 150)       // fail-high score blended towards beta, x% of 1/(depth+1)

// ---- Search audit additions (Oct 2026) -------------------------------------
// Standard techniques of modern engines (ideas as in Stockfish's search; own
// implementation and values). All off (0) by default; -DSR_NEW=1 turns on the
// whole set with first-guess values for a bundle test, before tuning.
// First tests (net 6, vs. the engine without them, @25 ms): all on -8 (600
// games, LOS 24%); by group (400 games each, +-25): LMR (ttPv, TT capture,
// cutoff count, capture LMR) +11, correction (minor, continuation) +10,
// qsearch futility +9, singular (triple, ttPv margin) +5, null move
// (verification, ttPv margin) +1, ordering (continuation 3/4/6, pawn
// history, capture history) -8; LMR + correction + qsearch + singular
// together -20 (600, LOS 2%; 23% fewer nodes at a fixed depth: likely too much
// pruning with untuned values). All off pending tuning (SPSA) and tests with
// more games / a longer time control (verification and triple extensions
// only act at depth >= 14 / deep singular searches).
#ifndef SR_NEW
#define SR_NEW 0
#endif
// ttPv: a node is "ttPv" if it is a PV node or its TT entry says it was on a
// PV once (TTEntry::pv). Such positions matter more: reduce them less.
SP_PARAM(TtPvLmrX100, SR_NEW ? 100 : 0, 0, 200)         // LMR -x/100 ply at ttPv nodes (non-PV; PV nodes already -1)
// Richer LMR (quiet moves).
SP_PARAM(LmrTtCapX100, SR_NEW ? 100 : 0, 0, 200)        // +x/100 ply when the TT move is a capture
SP_PARAM(LmrCutoffCntX100, SR_NEW ? 100 : 0, 0, 200)    // +x/100 ply when this node's children cut off > 2 times
// LMR for captures (not the TT move), reduced less with good capture history.
SP_PARAM(CapLmrOn, SR_NEW ? 1 : 0, 0, 1)
SP_PARAM(CapLmrBaseX100, 100, 0, 300)                     // captures: quiet reduction - x/100 ply
SP_PARAM(CapLmrHistDiv, 4096, 1024, 16384)               // capture history units per ply
// Capture history weighted like material (order += hist * CaptHistW / 256; the
// old CaptHistW range ended at 64) and used in SEE pruning of captures.
SP_PARAM(CaptHistOrdW, SR_NEW ? 128 : 0, 0, 256)          // overrides CaptHistW when > 0
SP_PARAM(CaptHistSeeDiv, SR_NEW ? 64 : 0, 0, 256)         // SEE prune threshold -= capture history / x (0 = off)
// More continuation history in move ordering (moves 3, 4 and 6 plies back,
// weight / 64) and pawn-structure history (quiet moves by pawn structure).
// Ordering only: LMR and history pruning keep the old history score.
SP_PARAM(ContOrdW3, SR_NEW ? 16 : 0, 0, 64)
SP_PARAM(ContOrdW4, SR_NEW ? 32 : 0, 0, 64)
SP_PARAM(ContOrdW6, SR_NEW ? 16 : 0, 0, 64)
SP_PARAM(PawnHistW, SR_NEW ? 64 : 0, 0, 128)
// Correction history by minor pieces (knights + bishops of both sides, kings)
// and by the previous move (continuation correction); weights / 1024.
SP_PARAM(CorrWMinor, SR_NEW ? 40 : 0, 0, 192)
SP_PARAM(CorrWCont, SR_NEW ? 40 : 0, 0, 192)
// Null move: verification search at high depth (null move off below the
// verification ply), and a margin above beta at ttPv nodes.
SP_PARAM(NmpVerifyDepth, SR_NEW ? 14 : 64, 6, 64)        // 64 = off
SP_PARAM(NmpTtPvMargin, SR_NEW ? 30 : 0, 0, 200)
// Singular extensions: triple extension far below singular beta (non-PV),
// and a wider singular margin at ttPv non-PV nodes.
SP_PARAM(TripleExtMargin, SR_NEW ? 120 : 0, 0, 400)       // 0 = off
SP_PARAM(SeTtPvX8, SR_NEW ? 8 : 0, 0, 32)
// Quiescence: a capture that doesn't win material (SEE < 1) is skipped when
// stand pat + margin is still at or below alpha.
SP_PARAM(QsFutMargin, SR_NEW ? 150 : 0, 0, 400)           // 0 = off

// ---- Efficiency audit switches (Oct 2026) ----------------------------------
// From the search efficiency audit (bench statistics against Stockfish 19,
// prototypes measured at fixed depth); the bench numbers are nodes at depth
// 13 against 9.07M. Screened one by one (1000 games each at 30 ms/move, all
// vs the defaults): LmrCutX100=100 +22, ChkLmr +19, CutLabelFix +12,
// QsMoveCount=2 +7, RfpBlend +6, the rest -12..0 (HistExactBonus -130 after
// 216 games). Those five together, with the parameters they touch re-tuned
// by SPSA at 10+0.1 (6000 pairs, preset tools/spsa/bundle on bundle_on:
// LmrCutX100 100 -> 85, LmrBase 0.91 -> 0.78, LmrDivisor 1.81 -> 1.89,
// HistLmrDivisor 4136 -> 6478, RfpMargin 63 -> 68, DeltaMargin 223 -> 248,
// IirDepth 4), vs the old defaults: SPRT [0, 5] H1 at 1+0.01, 5+0.05 and
// 10+0.1; 20+0.2 non-regression [-3, 2] stopped at LLR +1.8.  ADOPTED
// The other switches stay 0.
// Node types: a later move searched without reduction passes "cut" to its
// child (the node has not cut off on its first move, so it is probably an
// all-node, whose children are cut nodes). Before, it passed !cut_node, and
// expected all-nodes failed high 51% of the time (Stockfish 19%); with this
// 22%, expected cut-nodes 81% -> 88%. 8.97M nodes. Retest the cut-node
// features (NmpCutOnly, LmrCutX100) on top of it.
SP_PARAM(CutLabelFix, 1, 0, 1)
// LMR also for quiet moves that give check. 19% of main-search nodes were in
// check (Stockfish 11%): checks were never reduced, never pruned and always
// extended. 7.44M nodes.
SP_PARAM(ChkLmr, 1, 0, 1)
// Move-loop pruning (futility, SEE, history) also when the side to move has
// only pawns; null move stays off there (zugzwang). 9 endgames at depth 18:
// 13.2M -> 8.35M nodes.
SP_PARAM(PawnEgPrune, 0, 0, 1)
// Null move only at expected cut nodes (the SR_NMP_CUT switch as a runtime
// parameter). We tried null move at 12% of main nodes, Stockfish at 3%.
SP_PARAM(NmpCutOnly, 0, 0, 1)
// Quiescence: the TT move first (8.95M nodes), and at most this many
// captures searched per node when not in check, recaptures and promotions
// excepted (0 = no limit; 2: 8.57M nodes).
SP_PARAM(QsTtMove, 0, 0, 1)
SP_PARAM(QsMoveCount, 2, 0, 8)
// Reverse futility pruning returns (eval + beta) / 2 instead of the eval.
SP_PARAM(RfpBlend, 1, 0, 1)
// Correction history also learns from exact-bound results below the eval
// (before: only lower bounds above it and upper bounds below it).
SP_PARAM(CorrExact, 0, 0, 1)
// Lazy SMP: helper threads keep their move-ordering tables from one move to
// the next, as the main thread does (before: fresh tables every move).
SP_PARAM(HelperTables, 0, 0, 1)
// Check extension only after a check that doesn't lose material (SEE >= 0);
// before, every position in check was extended.
SP_PARAM(CheckExtSee, 0, 0, 1)
// History also learns from PV nodes: the best quiet move of a node that
// raised alpha without a cutoff gets the cutoff bonus (and the quiets tried
// before it the penalty). Before: only beta cutoffs taught the history.
// 1 = original version (bug: penalized every other quiet tried at the node,
//     including the ones searched after the best move, so every PV node,
//     the root included, pushed ~all its quiets negative; at depth 13
//     history pruning went 1.14M -> 4.28M moves, which explains the -130
//     screen). 2 = fixed: only quiets tried before the best move are
//     penalized, as on a cutoff (history pruning 1.64M). 3 = 2, not at root.
SP_PARAM(HistExactBonus, 0, 0, 3)
// Quiet moves tried before a capture that cut off get the history penalty.
SP_PARAM(CaptCutMalus, 0, 0, 1)
// Root moves after the previous best one in order of the nodes they took in
// the previous iteration (before: capture / history order).
SP_PARAM(RootOrder, 0, 0, 1)
// Aspiration window: start at AspDelta plus half the average change of the
// score between iterations (before: AspDelta alone; it failed about once per
// iteration).
SP_PARAM(AspAdaptive, 0, 0, 1)
#if SP_TUNE
// The known-ending rules of Eval.h (EV_KXK) as a tunable switch.
inline const bool KxkEval_registered = tune_register("KxkEval", &eval::g_kxk, EV_KXK, 0, 1);
#endif

// ---- Ordering experiments (Oct 2026), off by default ------------------------
// IID: at a PV / cut node without a TT move and depth >= IidDepth, first search
// the node IidReduction plies shallower to find a move to try first (instead
// of IIR's plain one-ply reduction there).
SP_PARAM(IidDepth, 0, 0, 20)          // 0 = off (IIR as before)
SP_PARAM(IidReduction, 4, 1, 8)
// Node types IID runs at (bits: 1 PV, 2 expected cut, 4 expected all; the
// default 3 = PV and cut nodes, as before). Without IID those nodes keep IIR.
SP_PARAM(IidNodes, 3, 1, 7)
// NNUE ordering: at depth >= NnueOrdDepth, the NnueOrdCount best quiets by
// history (0 = all of them) are each made, evaluated by the net from our side
// and put back in order of that eval, still ahead of the other quiets.
SP_PARAM(NnueOrdDepth, 0, 0, 20)      // 0 = off
SP_PARAM(NnueOrdCount, 4, 0, 32)
// Node types NNUE ordering runs at (bits as IidNodes; 7 = all, as before).
// By node type (Oct 5, new defaults, orderstats depth 13; base 7.11M nodes,
// 4.70 s, mean cutoff position 1.521, quiets 4.29): IID 8/4 at PV nodes
// 7.55M / 1.539 / 4.35, cut 8.11M / 1.534 / 4.17, all 7.28M / 1.544 / 4.42,
// PV + cut 8.17M / 1.532 / 4.20; NNUE ordering 6/4 at PV nodes 6.58M, 4.47 s /
// 1.522 / 4.45, cut 7.19M / 1.528 / 4.31, all 7.08M / 1.550 / 4.50, every
// node 7.43M / 1.542 / 4.44. No better ordering anywhere; NNUE ordering at PV
// nodes only cost no nodes (needs games).
SP_PARAM(NnueOrdNodes, 7, 1, 7)
inline constexpr int OrderNnueBand = 600'000;  // between counter moves and history quiets
// Measured (orderstats, depth 13; base: mean cutoff position 1.474, quiet
// moves 4.09, 9.07M nodes): IID 8/4 1.475 / 4.02 / 9.11M (only 41% of the IID
// searches leave a move: at cut nodes they mostly fail low), 6/3 1.463 / 4.01
// / 9.54M, 10/4 1.494 / 4.08 / 9.38M. NNUE ordering depth 6, 4 moves 1.476 /
// 3.99 / 8.79M; depth 6, 8 moves 1.477 / 4.11 / 9.40M; depth 4, all quiets
// 1.471 / 4.16 / 9.49M and 24% slower; depth 10, all 1.478 / 4.01 / 9.55M.
// Neither orders quiets better (OrdCheckBonus + OrdThreatX: 2.85 at no cost);
// both stay off.
inline int div_round(int a, int b) { return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b); }

// ---- Speed round (NPS work). Measured as instructions per node (callgrind,
// bench depth 9, start-up excluded) so machine noise doesn't hide small wins:
//   baseline                                              6620
//   SP_TT_EVAL: static eval cached in TT entries            6004  (-9.3%)
//   SP_STAGED: TT move searched before move generation      5870  (-2.2%)
//   SP_PAWN_HASH (Eval.h): pawn-only terms cached           5553  (-6.9%)
//   move scores not zero-filled; gives_check() only when
//   a pruning rule would fire                               5370  (-3.3%)
//   total                                                  -18.9%
// Everything but SP_STAGED (which orders quiets with fresher killers /
// history) gives identical node counts. Not kept: lazy qsearch eval (SP_LAZY,
// +4.7% nodes), lazy SEE in the picker (+0.2%), sorting the list after N
// picks (SP_SORT_AFTER, nodes +2-3%).
// A/B vs. the engine before, 800 games @25 ms: +304 -221 =275 (~+36 Elo).
// Eval-only TT entries (SP_TT_EVAL=2, -2% instructions), tested later @100 ms
// vs. the SPSA-tuned engine: +175 -135 =290 (600, ~+23, LOS ~99%): ON.
//
// Round 3: switches that were off, re-tested @100 ms/move vs. the speed +
// SPSA engine (300-game screens, adjudicated): mate distance pruning +6,
// null move only at cut nodes -8, ProbCut -14, eval-only TT entries +31,
// LMR +1 at cut nodes -2, eval-based null-move R -16, RFP by improving +19,
// LMR by improving -6, NMP by improving -3, richer LMR +23 (then -9: ~+7 over
// 600, off), capture history +8, ordering signals -13, pawn-push extension
// +10, fail-high blend +2. Adopted: eval-only TT entries + RFP by improving
// (each ~+21..23 over 600 games; together +179 -149 =272, 600, ~+17).
#ifndef SP_TT_EVAL
#define SP_TT_EVAL 2        // reuse the raw static eval cached in the TT entry
#endif                      // (2: also store an eval-only entry when the node had none)
inline constexpr bool UseTtEval = SP_TT_EVAL >= 1;
inline constexpr bool UseTtEval2 = SP_TT_EVAL >= 2;
#ifndef SP_STAGED
#define SP_STAGED 1         // search a legal TT move before generating the other moves
#endif
inline constexpr bool UseStaged = SP_STAGED;
#ifndef SP_LAZY
#define SP_LAZY 0           // qsearch lazy eval: material+PST only when that is SP_LAZY_MARGIN outside the window
#endif                      // tested: -1.8% instructions/node but +4.7% nodes to the same depth: off
#ifndef SP_LAZY_MARGIN
#define SP_LAZY_MARGIN 500  // full eval - fast eval: 99% within 360 cp, 99.9% within 480 (SF-labelled sets)
#endif
inline constexpr bool UseLazy = SP_LAZY;
// Also tried: SEE-testing captures only when picked (lazy SEE in the move
// picker): same order, but +0.2% instructions/node (most captures that are
// scored get picked anyway). Not kept.
inline constexpr int LazyMargin = SP_LAZY_MARGIN;

// ProbCut: at depth >= ProbCutMinDepth, a capture that beats
// beta + ProbCutMargin (- ProbCutImproving if improving) in a search
// reduced by ProbCutReduction probably beats beta at full depth too.
inline constexpr int ProbCutMinDepth  = 5;
inline constexpr int ProbCutReduction = 4;
inline constexpr int ProbCutMargin    = 200;
inline constexpr int ProbCutImproving = 50;

// Delta pruning (quiescence search): skip a capture if stand pat + the
// captured piece's value + DeltaMargin still can't reach alpha.
SP_PARAM(DeltaMargin, 248, 50, 500)

inline constexpr int LmrMinDepth = 3;
// Late move reductions: from this depth, quiet moves after the first
// LmrMinMoves moves are reduced by the log formula below.
SP_PARAM(LmrMinMoves, 2, 1, 6)
SP_PARAM(LmrBaseX100, 78, 0, 200)      // reduction = base + ln(depth) * ln(move number) / divisor
SP_PARAM(LmrDivisorX100, 189, 120, 400)

inline std::array<std::array<int, 64>, 64> build_lmr_table() {
    std::array<std::array<int, 64>, 64> t{};
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            t[static_cast<std::size_t>(d)][static_cast<std::size_t>(m)] =
                static_cast<int>(LmrBaseX100 / 100.0 + std::log(d) * std::log(m) / (LmrDivisorX100 / 100.0));
    return t;
}
inline std::array<std::array<int, 64>, 64>& lmr_table() {
    static std::array<std::array<int, 64>, 64> t = build_lmr_table();
    return t;
}
inline int lmr_reduction(int depth, int move_number) {
    return lmr_table()[static_cast<std::size_t>(std::min(depth, 63))][static_cast<std::size_t>(std::min(move_number, 63))];
}
// After a tuner changes parameters (SP_TUNE builds).
inline void tune_apply() { lmr_table() = build_lmr_table(); }

// One completed MultiPV line at some depth.
struct Line {
    Score score = 0;
    std::vector<chess::Move> pv;
};

// How the search talks to the outside world (the UCI layer).
struct Hooks {
    // Hard stop, polled every 2048 nodes: GUI stop, node limit, hard time.
    std::function<bool(std::uint64_t nodes)> should_stop;
    // Soft stop, checked before each new depth: depth limit, soft time.
    std::function<bool(int depth)> start_next_iteration;
    // Called once per completed line: depth, seldepth, multipv index (1-based),
    // line, total nodes, hashfull (permille).
    std::function<void(int, int, int, const Line&, std::uint64_t, int)> report;
};

struct Result {
    chess::Move best = chess::Move(chess::Move::NO_MOVE);
    chess::Move ponder = chess::Move(chess::Move::NO_MOVE);
};

class Searcher {
public:
    // `tables`: move-ordering statistics to use and keep updating (the UCI
    // layer passes its own so they carry over between moves); nullptr = fresh.
    Searcher(const chess::Board& board, TranspositionTable& tt, Hooks hooks, OrderTables* tables = nullptr)
        : board_(board), tt_(tt), hooks_(std::move(hooks)),
          own_tables_(tables ? nullptr : std::make_unique<OrderTables>()),
          ot_(tables ? *tables : *own_tables_) {
        moved_.fill(-1);
        tt_.set_buckets(TtBuckets != 0);
#if EV_NNUE
        board_.lazy_begin();
#endif
    }

    // Iterative deepening at the root.
    // `allowed` : root moves to consider (UCI searchmoves); empty = all legal.
    // `multipv` : number of best lines to find at each depth.
    Result iterate(const std::vector<chess::Move>& allowed, int multipv) {
        Result result;

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        for (const auto& m : legal)
            if (allowed.empty() || std::find(allowed.begin(), allowed.end(), m) != allowed.end())
                root_moves_.push_back(m);

        if (root_moves_.empty()) return result;  // mate / stalemate

        // Fallback so we always return a legal move, even if stopped at depth 1.
        result.best = root_moves_.front();
        const int lines = std::clamp(multipv, 1, static_cast<int>(root_moves_.size()));

        for (int depth = 1; depth < MaxPly && hooks_.start_next_iteration(depth); ++depth) {
            root_depth_ = depth;
            if (thread_id_ > 0 && depth > 1 && (depth + thread_id_) % 2 == 0) continue;
            if (!cur_root_nodes_.empty()) prev_root_nodes_.swap(cur_root_nodes_);
            cur_root_nodes_.clear();
            excluded_.clear();
            std::vector<Line> completed;

            // MultiPV: search the root once per line, excluding the best moves
            // already found at this depth.
            for (int k = 1; k <= lines; ++k) {
                seldepth_ = 0;
                Score score;
                if (UseAsp && k == 1 && lines == 1 && depth >= 4 && have_prev_ && !is_mate_score(prev_score_)) {
                    // Aspiration window around the last score, widened on a fail.
                    Score delta = AspDelta + (AspAdaptive ? asp_vol_ / 2 : 0);
                    Score a = std::max(prev_score_ - delta, -Infinite), b = std::min(prev_score_ + delta, Infinite);
                    for (;;) {
                        score = pvs(depth, 0, a, b, false, false);
                        if (stopped_) break;
                        if (score <= a) { b = (a + b) / 2; a = std::max(score - delta, -Infinite); }
                        else if (score >= b) { b = std::min(score + delta, Infinite); }
                        else break;
                        delta += delta / 2;
                        if (delta > 600) { a = -Infinite; b = Infinite; }
                    }
                } else {
                    score = pvs(depth, 0, -Infinite, Infinite, false, false);
                }
                if (stopped_) break;  // partial result: discard
                if (k == 1) {
                    if (have_prev_ && !is_mate_score(score) && !is_mate_score(prev_score_))
                        asp_vol_ = (3 * asp_vol_ + std::min(std::abs(score - prev_score_), 400)) / 4;
                    prev_score_ = score;
                    have_prev_ = true;
                }

                Line line{score, {pv_[0].begin(), pv_[0].begin() + pv_len_[0]}};
                excluded_.push_back(line.pv.front());
                hooks_.report(depth, seldepth_, k, line, nodes_, tt_.hashfull());
                completed.push_back(std::move(line));
            }

            // Line 1 is the best move. Use it if it finished, even if a later
            // MultiPV line of this depth was cut off.
            if (!completed.empty()) {
                const auto& pv = completed.front().pv;
                result.best = pv[0];
                pv_move_ = pv[0];  // searched first at the root next iteration
                result.ponder = pv.size() > 1 ? pv[1] : chess::Move(chess::Move::NO_MOVE);
            }
            if (stopped_) break;
        }
        return result;
    }

    std::uint64_t nodes() const { return nodes_; }

    // Lazy SMP: helper threads (id >= 1) search the same position with a
    // shared TT; they skip alternate depths so the threads spread out, and
    // their results reach the main thread (id 0) only through the TT.
    void set_thread_id(int id) { thread_id_ = id; }

private:
    // Principal Variation Search.
    // First move: full window. Remaining moves: null window (alpha, alpha+1)
    // to prove they are no better; re-search with the full window if one is.
    // `null_ok`: false right after a null move (no two null moves in a row).
    // `cut_node`: a non-PV node expected to fail high (its parent expects a
    // refutation here); the alternative non-PV kind is an all node.
    Score pvs(int depth, int ply, Score alpha, Score beta, bool null_ok, bool cut_node) {
        pv_len_[ply] = ply;

        // Check extension: never stop searching while in check.
        // In check: the parent usually knows already (known_check_, -1 = not known).
        const bool in_check = known_check_[static_cast<std::size_t>(ply)] >= 0 ? known_check_[static_cast<std::size_t>(ply)] != 0
                                                                               : board_.inCheck();
        known_check_[static_cast<std::size_t>(ply)] = in_check ? 1 : 0;  // for qsearch / razoring at this ply
        if (in_check && UseCheckExt && (!CheckExtSee || ext_ok_[static_cast<std::size_t>(ply)])) ++depth;

        if (depth <= 0) return qsearch(ply, alpha, beta);

        if (check_time()) return 0;
        SEARCH_STAT(++g_stats.nodes_main);
        seldepth_ = std::max(seldepth_, ply);

        if (ply > 0 && is_draw(ply)) return 0;
        if (ply >= MaxPly - 1) return eval::evaluate(board_);

        // Mate distance pruning: no line from here can beat a shorter mate
        // already found (or avoid a quicker loss).
        if (UseMdp && ply > 0) {
            alpha = std::max(alpha, static_cast<Score>(-Mate + ply));
            beta = std::min(beta, static_cast<Score>(Mate - ply - 1));
            if (alpha >= beta) return alpha;
        }

        const bool pv_node = (beta - alpha > 1);
        const Score alpha_orig = alpha;
        const std::uint64_t key = board_.hash();

        // TT probe. Cut off only in non-PV nodes so the PV stays complete.
        chess::Move tt_move(chess::Move::NO_MOVE);
        // Singular-extension verification search: this node is being searched
        // without the move in se_excluded_[ply]; the TT result for the node
        // (which includes that move) must not cut it off or be overwritten.
        const chess::Move se_excluded = se_excluded_[static_cast<std::size_t>(ply)];
        const bool se_search = se_excluded != chess::Move::NO_MOVE;
        TTEntry entry;
        const bool tt_found = UseTtEval2 ? tt_.probe_any(key, entry) : tt_.probe(key, entry);
        const bool tt_hit = tt_found && entry.bound != Bound::None;
        if (g_collect_stats && !se_search) {
            ++g_stats.tt_probes;
            g_stats.tt_found += tt_found;
            g_stats.tt_hit += tt_hit;
            g_stats.tt_move += tt_found && entry.move != 0;
        }
        if (tt_hit) {
            tt_move = chess::Move(entry.move);
            if (!pv_node && !se_search && entry.depth >= depth) {
                const Score s = score_from_tt(entry.score, ply);
                if (entry.bound == Bound::Exact ||
                    (entry.bound == Bound::Lower && s >= beta) ||
                    (entry.bound == Bound::Upper && s <= alpha))
                    return s;
            }
        }

        // ttPv: on a PV now, or the TT says this position was on one.
        const bool tt_pv = pv_node || (tt_found && entry.pv);
        const int node_bits = pv_node ? 1 : cut_node ? 2 : 4;  // IidNodes / NnueOrdNodes

        // Static evaluation, used by the pruning below. Not meaningful in check.
        // With SR_CORR it is corrected by what the search found in similar
        // positions (same pawns / same pieces).
        CorrKeys ck{};
        if (UseCorr) ck = corr_keys(ply);
        // The TT caches the eval before fifty-move scaling (the halfmove clock
        // isn't part of the key); the scaling is reapplied here.
        const int tt_store_eval = in_check ? TTNoEval
                                  : (UseTtEval && tt_found && entry.eval != TTNoEval) ? entry.eval
                                                                                   : eval::evaluate_unscaled(board_);
        const Score raw_eval = in_check ? -Infinite : fifty(tt_store_eval);
        if (UseTtEval2 && !tt_found && !in_check && !se_search)
            tt_.store(key, chess::Move(chess::Move::NO_MOVE), 0, 0, Bound::None, tt_store_eval);
        const Score static_eval = (UseCorr && !in_check && !eval::is_known_ending(board_)) ? corrected(raw_eval, ck) : raw_eval;

        // Improving: is our eval better than on our previous move (two plies
        // ago)? Falls back to four plies if that position was in check.
        eval_stack_[static_cast<std::size_t>(ply)] = in_check ? NoEval : static_eval;
        bool improving = false;
        if (!in_check) {
            if (ply >= 2 && eval_stack_[static_cast<std::size_t>(ply - 2)] != NoEval)
                improving = static_eval > eval_stack_[static_cast<std::size_t>(ply - 2)];
            else if (ply >= 4 && eval_stack_[static_cast<std::size_t>(ply - 4)] != NoEval)
                improving = static_eval > eval_stack_[static_cast<std::size_t>(ply - 4)];
            else
                improving = true;
        }

        if (!pv_node && !in_check) {
            // Reverse futility pruning (static null move): at shallow depth,
            // if we are so far above beta that even losing margin * depth
            // wouldn't bring us back under it, assume the node fails high.
            // When improving, a smaller margin is enough.
            const int rfp_depth = depth - (UseImprovingRfp && improving ? 1 : 0);
            if (depth <= RfpMaxDepth && !is_mate_score(beta) &&
                static_eval - RfpMargin * rfp_depth >= beta) {
                SEARCH_STAT(++g_stats.rfp);
                return RfpBlend ? (static_eval + beta) / 2 : static_eval;
            }

            // Razoring: far below alpha at low depth: see if captures can save it.
            if (UseRazor && depth <= 3 && !is_mate_score(alpha) && static_eval + RazorMargin * depth < alpha) {
                const Score v = qsearch(ply, alpha, beta);
                if (stopped_) return 0;
                if (v <= alpha) {
                    SEARCH_STAT(++g_stats.razor);
                    return v;
                }
            }

            // Null-move pruning: give the opponent a free move. If a reduced
            // search still fails high, a real move would too. Skipped without
            // pieces (pawn endgames), where zugzwang makes passing an
            // advantage and the assumption breaks.
            const bool nmp_eval_ok = static_eval >= beta && (!(UseNmpCut || NmpCutOnly) || cut_node) &&
                                     (!tt_pv || static_eval >= beta + NmpTtPvMargin);
            if (null_ok && !se_search && ply >= nmp_min_ply_ && depth >= NmpMinDepth && nmp_eval_ok && !is_mate_score(beta) &&
                board_.hasNonPawnMaterial(board_.sideToMove())) {
                const int r = NmpBase + depth / NmpDiv +
                              (NmpEvalScale > 0 ? std::min((static_eval - beta) * NmpEvalScale / 20000, 3) : 0);
                SEARCH_STAT(++g_stats.nmp);
                moved_[static_cast<std::size_t>(ply)] = -1;
                known_check_[static_cast<std::size_t>(ply + 1)] = 0;  // a null move never gives check
                board_.makeNullMove();
                // The null move's reply node: the opposite of this node's type
                // (after a cut node an all node is expected and vice versa). Was
                // always "not cut" before the Oct 2026 audit; no effect at the
                // depth of 25 ms tests (identical bench), visible from ~depth 15.
                const Score score = -pvs(depth - 1 - r, ply + 1, -beta, -beta + 1, false, !cut_node);
                board_.unmakeNullMove();
                if (stopped_) return 0;
                if (score >= beta) {
                    SEARCH_STAT(++g_stats.nmp_cut);
                    // At high depth, verify with a normal reduced search of this
                    // node with null moves off for the next plies (zugzwang).
                    if (depth < NmpVerifyDepth || nmp_min_ply_ != 0)
                        return is_mate_score(score) ? beta : score;  // don't trust null-move mates
                    nmp_min_ply_ = ply + 3 * (depth - 1 - r) / 4;
                    const Score v = pvs(depth - 1 - r, ply, beta - 1, beta, false, cut_node);
                    nmp_min_ply_ = 0;
                    if (stopped_) return 0;
                    if (v >= beta) return is_mate_score(score) ? beta : score;
                }
            }

            // ProbCut: if a good capture beats a raised beta in a much
            // shallower search, it very likely beats beta at full depth.
            if (ProbCutStr > 0 && !se_search && depth >= ProbCutMinDepth && !is_mate_score(beta)) {
                const Score pc_beta = beta + (1000 - 8 * ProbCutStr) - (improving ? ProbCutImproving : 0);
                // Skip if the TT already says this position is below pc_beta.
                const bool tt_says_low = tt_hit && entry.depth >= depth - ProbCutReduction + 1 &&
                                         score_from_tt(entry.score, ply) < pc_beta;
                if (!tt_says_low) {
                    SEARCH_STAT(++g_stats.pc);
                    OrderedMoves caps;
                    chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(caps.moves, board_);
                    score_moves(caps, tt_move, ply);
                    for (int i = 0; i < caps.size(); ++i) {
                        const chess::Move move = caps.next(i);
                        if (!see_ge(board_, move, pc_beta - static_eval)) continue;
                        known_check_[static_cast<std::size_t>(ply + 1)] = -1;
                        ext_ok_[static_cast<std::size_t>(ply + 1)] = true;
                        make(move, ply);
                        Score score = -qsearch(ply + 1, -pc_beta, -pc_beta + 1);
                        // Verification: the reply searched ProbCutReduction plies
                        // shallower than this node (child depth - 4). Before the
                        // Oct 2026 audit it was depth - 1 - 4: one ply too shallow,
                        // so at depth 5 it repeated the qsearch above.
                        if (score >= pc_beta)
                            score = -pvs(depth - ProbCutReduction, ply + 1, -pc_beta, -pc_beta + 1, true, !cut_node);
                        board_.unmake(move);
                        if (stopped_) return 0;
                        if (score >= pc_beta) {
                            SEARCH_STAT(++g_stats.pc_cut);
                            tt_.store(key, move, score_to_tt(score, ply), depth - ProbCutReduction + 1, Bound::Lower,
                                      tt_store_eval, tt_pv);
                            return score;
                        }
                    }
                }
            }
        }

        // PV move: at the root, the previous iteration's best move is
        // searched first even if its TT entry has been overwritten.
        if (ply == 0 && tt_move == chess::Move::NO_MOVE) tt_move = pv_move_;

        // Internal iterative reduction: a PV / cut node without a TT move is
        // probably badly ordered; search it one ply shallower.
        if (IidDepth > 0 && (node_bits & IidNodes) && depth >= IidDepth && tt_move == chess::Move::NO_MOVE && !se_search) {
            // IID: a shallower search of this node (null move off: already tried here)
            // leaves its best move in the TT.
            pvs(depth - IidReduction, ply, alpha, beta, false, cut_node);
            if (stopped_) return 0;
            TTEntry iid;
            if (tt_.probe(key, iid)) tt_move = chess::Move(iid.move);
            SEARCH_STAT(++g_stats.iid; g_stats.iid_move += tt_move != chess::Move::NO_MOVE);
        } else if (UseIir && (pv_node || cut_node) && depth >= IirDepth &&
                   (tt_move == chess::Move::NO_MOVE || (FailLowIir && tt_hit && entry.bound == Bound::Upper))) {
            --depth;
        }

        // Staged: a legal TT move (not at the root, not in a singular
        // verification search) is searched before the move list is built;
        // the list is only generated if it doesn't cut off. The list order is
        // the same either way (the TT move scores highest and comes first).
        OrderedMoves list;
        const bool tt_first = UseStaged && ply > 0 && !se_search && is_legal_move(board_, tt_move);
        bool generated = false;
        auto generate = [&]() {
            chess::movegen::legalmoves(list.moves, board_);
            score_moves(list, tt_move, ply);
            if (NnueOrdDepth > 0 && depth >= NnueOrdDepth && !in_check && (node_bits & NnueOrdNodes)) nnue_reorder(list);
            generated = true;
        };
        if (!tt_first) {
            generate();
            if (list.moves.empty())
                return board_.inCheck() ? -Mate + ply : 0;  // checkmate : stalemate
            if (se_search && list.size() == 1) return alpha;  // only the excluded move: nothing to compare
        }

        Score best = -Infinite;
        chess::Move best_move(chess::Move::NO_MOVE);
        chess::Move fl_move(chess::Move::NO_MOVE);  // best-scoring move, even below alpha (FailLowMove)
        int best_quiet_idx = 0;  // quiets tried before best_move (HistExactBonus >= 2)
        int moves_searched = 0;
        // Beta cutoffs among this node's children (LmrCutoffCnt).
        cutoff_cnt_[static_cast<std::size_t>(ply + 1)] = 0;
        const bool tt_capture = tt_move != chess::Move::NO_MOVE && board_.isCapture(tt_move);

        // Quiet moves searched so far that didn't cause a cutoff: they get a
        // history penalty if a later quiet move does.
        std::array<chess::Move, 64> quiets_tried;
        int quiet_count = 0;
        std::array<chess::Move, 32> captures_tried;
        int capture_count = 0;

        for (int i = 0;; ++i) {
            chess::Move move;
            if (tt_first && i == 0) {
                move = tt_move;
            } else {
                if (!generated) {
                    generate();
                    list.next(0);  // the TT move, already searched
                }
                if (i >= list.size()) break;
                move = list.next(i);
            }
            if (ply == 0 && !root_move_allowed(move)) continue;
            if (move == se_excluded) continue;
            const bool quiet = is_quiet(board_, move);

            // Extensions, limited so the search can't run away (ply < 2 x root depth).
            int extension = 0;
            if (ply > 0 && ply < 2 * root_depth_) {
                if (UseSingular && move == tt_move && !se_search && depth >= SingularMinDepth && tt_hit &&
                    entry.bound != Bound::Upper && entry.depth >= depth - 3 &&
                    !is_mate_score(score_from_tt(entry.score, ply))) {
                    const Score s_beta = score_from_tt(entry.score, ply) -
                                         (SingularMarginX8 + (tt_pv && !pv_node ? SeTtPvX8 : 0)) * depth / 8;
                    se_excluded_[static_cast<std::size_t>(ply)] = move;
                    const Score s = pvs((depth - 1) / 2, ply, s_beta - 1, s_beta, false, cut_node);
                    se_excluded_[static_cast<std::size_t>(ply)] = chess::Move(chess::Move::NO_MOVE);
                    if (stopped_) return 0;
                    SEARCH_STAT(++g_stats.se; g_stats.se_ext += s < s_beta; g_stats.se_multicut += s >= s_beta && s_beta >= beta);
                    if (s < s_beta) {                    // only the TT move is good: search it deeper
                        extension = 1;
                        if (UseSeDouble && !pv_node && s < s_beta - DoubleExtMargin && double_ext_[static_cast<std::size_t>(ply)] < 6) extension = 2;
                        if (TripleExtMargin > 0 && extension == 2 && s < s_beta - TripleExtMargin) extension = 3;
                    }
                    else if (s_beta >= beta) return s_beta;  // several moves beat beta: multi-cut
                    else if (UseSeNeg && (score_from_tt(entry.score, ply) >= beta || cut_node)) extension = -1;
                }
                if (UsePawnExt && extension == 0 && board_.at(move.from()).type() == chess::PieceType::PAWN &&
                    move.typeOf() != chess::Move::PROMOTION) {
                    const int to_rank = move.to().index() / 8;
                    if ((board_.sideToMove() == chess::Color::WHITE ? to_rank : 7 - to_rank) == 6) extension = 1;
                }
            }

            // Late move pruning: at shallow depth, once enough moves have been
            // searched, skip the remaining quiet moves (ordered last, rarely
            // best). Fewer are searched when not improving.
            const bool lmp_skip = UseLmp && ply > 0 && !in_check && quiet && depth <= LmpMaxDepth &&
                                  best > -(Mate - MaxPly) && moves_searched >= lmp_limit(depth, improving);
            if (lmp_skip && !LmpKeepChecks) {
                SEARCH_STAT(++g_stats.lmp);
                continue;
            }

            const int move_hist = quiet ? quiet_history(move, ply) : 0;  // before the move is made
            const int cap_hist = (!quiet && is_capture_stat(move) && capt_hist_on()) ? ot_.capture_get(board_, move) : 0;

            // Move-loop pruning (after the first move): futility, history and
            // SEE pruning of quiets, SEE pruning of captures.
            if (UseMovePruning && ply > 0 && !in_check && best > -(Mate - MaxPly) && move != tt_move &&
                (PawnEgPrune || board_.hasNonPawnMaterial(board_.sideToMove()))) {
                if (quiet) {
                    // Checking quiets are never pruned; gives_check() is only
                    // asked once a rule would prune the move.
                    const int lmr_d = std::max(0, depth - 1 - lmr_reduction(depth, moves_searched + 1));
                    const bool p_hist = UseHistPrune && depth <= HistPruneDepth && move_hist < -HistPruneMul * depth;
                    const bool p_fut = !p_hist && UseFutPrune && lmr_d <= FutDepth && static_eval + FutBase + FutMul * lmr_d <= alpha;
                    const bool p_see = !p_hist && !p_fut && UseSeePrune && !see_ge(board_, move, -SeeQuietMul * lmr_d * lmr_d);
                    if ((p_hist || p_fut || p_see) && !gives_check(board_, move)) {
                        SEARCH_STAT(g_stats.hist += p_hist; g_stats.fut += p_fut; g_stats.see_q += p_see);
                        continue;
                    }
                } else if (UseSeePrune && depth <= 8 &&
                           !see_ge(board_, move, -SeeCapMul * depth - (CaptHistSeeDiv > 0 ? cap_hist / CaptHistSeeDiv : 0))) {
                    SEARCH_STAT(++g_stats.see_c);
                    continue;
                }
            }
            const bool bad_capture = !quiet && generated && list.scores[static_cast<std::size_t>(i)] < 0;
            const bool refuter = quiet && (move == killers_[static_cast<std::size_t>(ply)][0] ||
                                           move == killers_[static_cast<std::size_t>(ply)][1]);
            ext_ok_[static_cast<std::size_t>(ply + 1)] =
                !CheckExtSee || !search::gives_check(board_, move) || see_ge(board_, move, 0);
            make(move, ply);
            const bool gives_check = board_.inCheck();
            known_check_[static_cast<std::size_t>(ply + 1)] = gives_check ? 1 : 0;
            if (lmp_skip && !gives_check) {  // (LmpKeepChecks) checks are still searched
                SEARCH_STAT(++g_stats.lmp);
                board_.unmake(move);
                continue;
            }
            const std::uint64_t nodes_before = nodes_;
            const int new_depth = depth - 1 + extension;
            double_ext_[static_cast<std::size_t>(ply + 1)] =
                double_ext_[static_cast<std::size_t>(ply)] + (extension >= 2 ? 1 : 0);
            Score score;
            if (moves_searched == 0) {
                score = -pvs(new_depth, ply + 1, -beta, -alpha, true, pv_node ? false : !cut_node);
            } else {
                // Late move reductions: well-ordered moves late in the list
                // rarely matter, so search them shallower first and only
                // re-search at full depth if they beat alpha.
                int r = 0;
                const bool cap_lmr = !quiet && ((LmrBadCapX100 > 0 && bad_capture) ||
                                                (CapLmrOn > 0 && move != tt_move && is_capture_stat(move)));
                if (depth >= LmrMinDepth && moves_searched >= LmrMinMoves && (quiet || cap_lmr) &&
                    !in_check && (!gives_check || (ChkLmr && quiet))) {
                    r = lmr_reduction(depth, moves_searched + 1);
                    if (pv_node) --r;
                    if (UseHistoryLmr && quiet) r -= move_hist / HistLmrDivisor;
                    // Fractional adjustments (x/100 ply), rounded once.
                    int adj = 0;
                    if (!improving) adj += LmrNotImpX100;
                    if (cut_node) adj += LmrCutX100;
                    if (refuter) adj -= LmrRefuterX100;
                    if (tt_pv && !pv_node) adj -= TtPvLmrX100;
                    if (tt_capture) adj += LmrTtCapX100;
                    if (cutoff_cnt_[static_cast<std::size_t>(ply + 1)] > 2) adj += LmrCutoffCntX100;
                    if (!quiet) {
                        if (CapLmrOn > 0) {  // captures: a bit less, and less with good capture history
                            adj -= CapLmrBaseX100;
                            r -= cap_hist / CapLmrHistDiv;
                        } else {
                            adj += r * (LmrBadCapX100 - 100);  // bad capture: x% of the reduction
                        }
                    }
                    if (adj != 0) r = div_round(r * 100 + adj, 100);
                    r = std::clamp(r, 0, new_depth - 1);
                }

                // Reduced searches expect a refutation (cut node); an unreduced
                // null-window search flips the parent's expectation.
                score = -pvs(new_depth - r, ply + 1, -alpha - 1, -alpha, true, (r > 0 || CutLabelFix) ? true : !cut_node);
                SEARCH_STAT(if (r > 0) { ++g_stats.lmr; g_stats.lmr_research += score > alpha; });
                if (score > alpha && r > 0) {  // reduced search surprised us
                    int nd = new_depth;
                    if (UseLmr2) nd += (score > best + 60 ? 1 : 0) - (score < best + 10 ? 1 : 0);
                    if (nd > new_depth - r) score = -pvs(nd, ply + 1, -alpha - 1, -alpha, true, !cut_node);
                }
                if (score > alpha && score < beta)  // PVS re-search, full window
                    score = -pvs(new_depth, ply + 1, -beta, -alpha, true, false);
            }
            board_.unmake(move);
            ++moves_searched;
            if (ply == 0) add_root_nodes(move, nodes_ - nodes_before);

            if (stopped_) return 0;

            if (score > best) {
                best = score;
                fl_move = move;
                if (score > alpha) {
                    alpha = score;
                    best_move = move;
                    best_quiet_idx = quiet_count;
                    update_pv(ply, move);
                    if (alpha >= beta) {  // fail high (beta cutoff)
                        if (g_collect_stats && ply > 0) record_cutoff(i, move, tt_move, list, generated, tt_first, pv_node);
                        ++cutoff_cnt_[static_cast<std::size_t>(ply)];
                        if (quiet) on_quiet_cutoff(move, ply, depth, quiets_tried, quiet_count);
                        else {
                            if (capt_hist_on() && is_capture_stat(move))
                                update_stat(ot_.capture_at(board_, move), history_bonus(depth));
                            if (CaptCutMalus)
                                for (int k = 0; k < quiet_count; ++k)
                                    update_quiet(quiets_tried[static_cast<std::size_t>(k)], ply, -history_bonus(depth));
                        }
                        if (capt_hist_on())  // captures that were tried first didn't work
                            for (int k = 0; k < capture_count; ++k)
                                update_stat(ot_.capture_at(board_, captures_tried[static_cast<std::size_t>(k)]),
                                            -history_bonus(depth));
                        break;
                    }
                }
            }
            if (quiet && quiet_count < static_cast<int>(quiets_tried.size()))
                quiets_tried[static_cast<std::size_t>(quiet_count++)] = move;
            else if (!quiet && is_capture_stat(move) && capture_count < static_cast<int>(captures_tried.size()))
                captures_tried[static_cast<std::size_t>(capture_count++)] = move;
        }

        // HistExactBonus: a quiet best move that raised alpha without a cutoff.
        if (HistExactBonus && !se_search && best_move != chess::Move::NO_MOVE && best < beta && is_quiet(board_, best_move) &&
            (HistExactBonus != 3 || ply > 0)) {
            std::array<chess::Move, 64> others;
            int n_others = 0;
            const int n_pen = HistExactBonus >= 2 ? best_quiet_idx : quiet_count;
            for (int k = 0; k < n_pen; ++k)
                if (quiets_tried[static_cast<std::size_t>(k)] != best_move) others[static_cast<std::size_t>(n_others++)] = quiets_tried[static_cast<std::size_t>(k)];
            on_quiet_cutoff(best_move, ply, depth, others, n_others);
        }

        if (FhBlendX > 0 && best >= beta && !is_mate_score(best) && !is_mate_score(beta))
            best -= (best - beta) * FhBlendX / (100 * (depth + 1));

        // Correction history: the search result vs. the static eval, when the
        // bound says which way the eval was wrong (and the best move is quiet).
        if (UseCorr && !in_check && !se_search && !is_mate_score(best) &&
            (best_move == chess::Move::NO_MOVE || is_quiet(board_, best_move)) &&
            (CorrExact ? !(best >= beta && best <= static_eval) && !(best <= alpha_orig && best >= static_eval)
                       : ((best_move != chess::Move::NO_MOVE) ? best > static_eval : best < static_eval))) {
            const int bonus = std::clamp((best - static_eval) * depth / 8, -CorrLimit / 4, CorrLimit / 4);
            update_corr(ck, bonus);
        }

        // Don't store the root while excluding MultiPV moves: that score is
        // for "best move except the ones already found", not the position.
        // PriorCmBonus: every move failed low, so the opponent's quiet move that
        // led here was good.
        if (PriorCmBonus > 0 && best <= alpha_orig && ply > 0 && !se_search && moves_searched > 0 &&
            played_quiet_[static_cast<std::size_t>(ply - 1)])
            prior_move_bonus(ply, history_bonus(depth) * PriorCmBonus / 100);

        if (!(ply == 0 && !excluded_.empty()) && !se_search) {
            const Bound bound = best >= beta        ? Bound::Lower
                              : best > alpha_orig   ? Bound::Exact
                                                    : Bound::Upper;
            chess::Move store_move = best_move;
            if (FailLowMove && store_move == chess::Move::NO_MOVE && (FailLowMove == 1 || tt_move == chess::Move::NO_MOVE))
                store_move = fl_move;
            tt_.store(key, store_move, score_to_tt(best, ply), depth, bound, tt_store_eval, tt_pv);
        }
        return best;
    }

    // NNUE ordering (NnueOrdDepth): the NnueOrdCount best history-ordered quiets
    // are re-ranked by the net's eval after each move, from our side.
    void nnue_reorder(OrderedMoves& list) {
        std::array<int, 256> idx;
        int n = 0;
        for (int i = 0; i < list.size(); ++i) {
            const int sc = list.scores[static_cast<std::size_t>(i)];
            if (sc < OrderCounter && sc > OrderBadCapture + 100'000 && is_quiet(board_, list.moves[i])) idx[static_cast<std::size_t>(n++)] = i;
        }
        if (n < 2) return;
        const int k = NnueOrdCount > 0 ? std::min(n, static_cast<int>(NnueOrdCount)) : n;
        auto by_score = [&](int a, int b) { return list.scores[static_cast<std::size_t>(a)] > list.scores[static_cast<std::size_t>(b)]; };
        std::partial_sort(idx.begin(), idx.begin() + k, idx.begin() + n, by_score);
        std::array<std::pair<int, int>, 256> ev;  // (eval after the move, our view; list index)
        for (int j = 0; j < k; ++j) {
            const chess::Move m = list.moves[idx[static_cast<std::size_t>(j)]];
            board_.make(m);
            ev[static_cast<std::size_t>(j)] = {-eval::evaluate(board_), idx[static_cast<std::size_t>(j)]};
            board_.unmake(m);
        }
        std::stable_sort(ev.begin(), ev.begin() + k, [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int j = 0; j < k; ++j) list.scores[static_cast<std::size_t>(ev[static_cast<std::size_t>(j)].second)] = OrderNnueBand + (k - j);
        SEARCH_STAT(++g_stats.nnue_ord; g_stats.nnue_evals += k);
    }

    // Category of a move by its order score (SearchStats).
    int order_category(int score, bool is_tt, bool quiet) const {
        if (is_tt) return 0;
        if (score >= OrderCapture) return 1;
        if (score == OrderKiller1) return 2;
        if (score == OrderKiller2) return 3;
        if (score == OrderCounter) return 4;
        if (score <= OrderUnderPromo + 100'000) return 7;
        if (score <= OrderBadCapture + 100'000 && !quiet) return 6;
        return 5;
    }
    void record_cutoff(int i, const chess::Move& move, const chess::Move& tt_move, const OrderedMoves& list,
                       bool generated, bool tt_first, bool pv_node) {
        SearchStats& st = g_stats;
        const int idx = i + 1;
        ++st.cutoffs;
        st.first += idx == 1;
        st.top3 += idx <= 3;
        st.idx_sum += idx;
        if (!pv_node) {
            ++st.nonpv;
            st.nonpv_first += idx == 1;
        }
        const int sc = generated ? list.scores[static_cast<std::size_t>(i)] : OrderHash;
        const int c = order_category(sc, move == tt_move, is_quiet(board_, move));
        ++st.cat[c];
        st.cat_idx[c] += idx;
        if (idx > 1) {
            ++st.late;
            ++st.late_cat[c];
            const bool first_tt = tt_first || (generated && list.moves[0] == tt_move);
            const int sc0 = generated ? list.scores[0] : OrderHash;
            ++st.fail_first_cat[order_category(sc0, first_tt, generated && is_quiet(board_, list.moves[0]))];
        }
    }

    Score fifty(int unscaled) const {
        return eval::fifty_move_scale(unscaled, static_cast<int>(board_.halfMoveClock()));
    }

    // Quiescence search: keep searching captures until the position is quiet,
    // so the static eval isn't taken in the middle of an exchange.
    Score qsearch(int ply, Score alpha, Score beta) {
        pv_len_[ply] = ply;

        if (check_time()) return 0;
        SEARCH_STAT(++g_stats.nodes_qs);
        seldepth_ = std::max(seldepth_, ply);

        if (is_draw(ply)) return 0;
        if (ply >= MaxPly - 1) return eval::evaluate(board_);

        const bool pv_node = (beta - alpha > 1);
        const Score alpha_orig = alpha;
        const std::uint64_t key = board_.hash();

        // Any stored result (depth >= 0) is at least as good as a qsearch.
        TTEntry entry;
        const bool tt_found = (UseTtEval || !pv_node) && (UseTtEval2 ? tt_.probe_any(key, entry) : tt_.probe(key, entry));
        const bool tt_hit = tt_found && entry.bound != Bound::None;
        const bool q_pv = pv_node || (tt_found && entry.pv);
        if (!pv_node && tt_hit) {
            const Score s = score_from_tt(entry.score, ply);
            if (entry.bound == Bound::Exact ||
                (entry.bound == Bound::Lower && s >= beta) ||
                (entry.bound == Bound::Upper && s <= alpha))
                return s;
        }

        const bool in_check = known_check_[static_cast<std::size_t>(ply)] >= 0 ? known_check_[static_cast<std::size_t>(ply)] != 0
                                                                               : board_.inCheck();
        OrderedMoves list;
        Score best;
        Score stand_pat = -Infinite;
        int raw_eval = TTNoEval;  // raw static eval, cached in the TT (not in check)

        if (in_check) {
            // No standing pat in check: every evasion must be tried.
            chess::movegen::legalmoves(list.moves, board_);
            if (list.moves.empty()) return -Mate + ply;
            best = -Infinite;
        } else {
            // Stand pat: the side to move can usually do at least as well as
            // the static eval by making a quiet move.
            const bool have_tt_eval = UseTtEval && tt_found && entry.eval != TTNoEval;
            bool lazy_low = false;
            if (UseLazy && !have_tt_eval && !is_mate_score(alpha) && !is_mate_score(beta)) {
                // Lazy eval: if material + PST alone is far outside the window,
                // the full eval (within LazyMargin of it) can't bring it back.
                const Score fast = fifty(eval::fast_eval(board_));
                if (fast - LazyMargin >= beta) return fast - LazyMargin;
                if (fast + LazyMargin <= alpha) {
                    best = stand_pat = fast + LazyMargin;  // an upper bound on stand pat
                    lazy_low = true;
                }
            }
            if (!lazy_low) {
                raw_eval = have_tt_eval ? entry.eval : eval::evaluate_unscaled(board_);
                const Score e = fifty(raw_eval);
                best = stand_pat = (UseCorr && !eval::is_known_ending(board_)) ? corrected(e, corr_keys(ply)) : e;
                if (best >= beta) {
                    tt_.store(key, chess::Move(chess::Move::NO_MOVE), score_to_tt(best, ply), 0,
                              Bound::Lower, raw_eval, q_pv);
                    return best;
                }
                alpha = std::max(alpha, best);
            }
            chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(list.moves, board_);
        }
        score_moves(list, (QsTtMove && tt_hit) ? chess::Move(entry.move) : chess::Move(chess::Move::NO_MOVE), ply);

        chess::Move best_move(chess::Move::NO_MOVE);
        int qs_searched = 0;
        // Square of the previous move's destination (recaptures are never cut by QsMoveCount).
        const int prev_to = ply > 0 && moved_[static_cast<std::size_t>(ply - 1)] >= 0 ? moved_[static_cast<std::size_t>(ply - 1)] % 64 : -1;
        for (int i = 0; i < list.size(); ++i) {
            const chess::Move move = list.next(i);
            if (QsMoveCount > 0 && !in_check && qs_searched >= QsMoveCount && move.to().index() != prev_to &&
                move.typeOf() != chess::Move::PROMOTION)
                continue;

            if (!in_check) {
                // Losing captures (SEE < 0) and under-promotions are ordered
                // last, with negative scores: from the first one on, stop.
                if (list.scores[static_cast<std::size_t>(i)] < 0) break;

                // Delta pruning: even winning the captured piece outright
                // (plus a safety margin) wouldn't lift us to alpha.
                if (move.typeOf() != chess::Move::PROMOTION) {
                    const Score gain = move.typeOf() == chess::Move::ENPASSANT
                                           ? eval::PawnValue
                                           : see_value(board_.at(move.to()).type());
                    if (stand_pat + gain + DeltaMargin <= alpha) continue;
                    // Futility: far enough below alpha that only winning material helps.
                    if (QsFutMargin > 0 && stand_pat + QsFutMargin <= alpha && !see_ge(board_, move, 1)) continue;
                }
            }

            ++qs_searched;
            known_check_[static_cast<std::size_t>(ply + 1)] = -1;
            make(move, ply);
            const Score score = -qsearch(ply + 1, -beta, -alpha);
            board_.unmake(move);

            if (stopped_) return 0;

            if (score > best) {
                best = score;
                if (score > alpha) {
                    alpha = score;
                    best_move = move;
                    if (alpha >= beta) break;
                }
            }
        }

        const Bound bound = best >= beta      ? Bound::Lower
                          : best > alpha_orig ? Bound::Exact
                                              : Bound::Upper;
        tt_.store(key, best_move, score_to_tt(best, ply), 0, bound, raw_eval, q_pv);
        return best;
    }

    // Assign order scores to every move in the list.
    void score_moves(OrderedMoves& list, const chess::Move& tt_move, int ply) const {
        const auto& killers = killers_[static_cast<std::size_t>(ply)];
        const Threats threats = OrdThreatX > 0 ? compute_threats() : Threats{};
        for (int i = 0; i < list.size(); ++i) {
            const chess::Move m = list.moves[i];
            int score;
            if (m == tt_move) {
                score = OrderHash;
            } else if (RootOrder && ply == 0 && !prev_root_nodes_.empty()) {
                score = root_order_score(m);
            } else if (m.typeOf() == chess::Move::PROMOTION) {
                const bool queen = m.promotionType() == chess::PieceType::QUEEN;
                const int capture = board_.isCapture(m) ? mvv_lva(board_, m) : 0;
                score = queen ? OrderCapture + eval::QueenValue * 8 + capture
                              : OrderUnderPromo + capture;
            } else if (!is_quiet(board_, m)) {
                const int base = see_ge(board_, m, 0) ? OrderCapture : OrderBadCapture;
                score = base + mvv_lva(board_, m);
                if (capt_hist_w() > 0) score += ot_.capture_get(board_, m) * capt_hist_w() / 256;
            } else if (m == killers[0]) {
                score = OrderKiller1;
            } else if (m == killers[1]) {
                score = OrderKiller2;
            } else if (UseCountermove && ply > 0 && moved_[static_cast<std::size_t>(ply - 1)] >= 0 &&
                       m == ot_.counter[static_cast<std::size_t>(moved_[static_cast<std::size_t>(ply - 1)])]) {
                score = OrderCounter;
            } else {
                score = order_history(m, ply);
                if (OrdCheckBonus > 0 || OrdThreatX > 0) score += ord2_bonus(m, threats);
            }
            list.scores[static_cast<std::size_t>(i)] = score;
        }
    }

    // SR_ORD2: squares attacked by enemy pieces of lower value than each of
    // our piece types (a piece standing there can be won).
    struct Threats {
        std::uint64_t by_lesser[6]{};
    };
    Threats compute_threats() const {
        using chess::PieceType;
        Threats t;
        const chess::Color them = ~board_.sideToMove();
        const chess::Bitboard occ = board_.occ();
        std::uint64_t pawn = 0, minor = 0, rook = 0;
        auto each = [&](PieceType pt, auto&& f) {
            for (std::uint64_t b = board_.pieces(pt, them).getBits(); b; b &= b - 1)
                f(chess::Square(static_cast<int>(std::countr_zero(b))));
        };
        each(PieceType::PAWN, [&](chess::Square sq) { pawn |= chess::attacks::pawn(them, sq).getBits(); });
        each(PieceType::KNIGHT, [&](chess::Square sq) { minor |= chess::attacks::knight(sq).getBits(); });
        each(PieceType::BISHOP, [&](chess::Square sq) { minor |= chess::attacks::bishop(sq, occ).getBits(); });
        each(PieceType::ROOK, [&](chess::Square sq) { rook |= chess::attacks::rook(sq, occ).getBits(); });
        t.by_lesser[1] = t.by_lesser[2] = pawn;
        t.by_lesser[3] = pawn | minor;
        t.by_lesser[4] = pawn | minor | rook;
        return t;
    }
    int ord2_bonus(const chess::Move& m, const Threats& t) const {
        int b = 0;
        if (OrdCheckBonus > 0 && gives_check(board_, m) && see_ge(board_, m, -75)) b += OrdCheckBonus;
        if (OrdThreatX <= 0) return b;
        const auto pt = board_.at(m.from()).type();
        const int idx = static_cast<int>(pt.internal());
        if (idx >= 1 && idx <= 4) {
            const std::uint64_t lesser = t.by_lesser[static_cast<std::size_t>(idx)];
            const int from_t = static_cast<int>((lesser >> m.from().index()) & 1);
            const int to_t = static_cast<int>((lesser >> m.to().index()) & 1);
            b += OrdThreatX * see_value(pt) * (from_t - to_t);
        }
        return b;
    }

    // A quiet move caused a beta cutoff: remember it as a killer for this
    // ply, reward it in the history table, and penalize the quiet moves
    // searched before it that failed to cut off.
    void on_quiet_cutoff(const chess::Move& move, int ply, int depth,
                         const std::array<chess::Move, 64>& tried, int tried_count) {
        auto& killers = killers_[static_cast<std::size_t>(ply)];
        if (killers[0] != move) {
            killers[1] = killers[0];
            killers[0] = move;
        }
        const int bonus = history_bonus(depth);
        update_quiet(move, ply, bonus);
        for (int i = 0; i < tried_count; ++i) update_quiet(tried[static_cast<std::size_t>(i)], ply, -bonus);
        if (UseCountermove && ply > 0 && moved_[static_cast<std::size_t>(ply - 1)] >= 0)
            ot_.counter[static_cast<std::size_t>(moved_[static_cast<std::size_t>(ply - 1)])] = move;
    }

    // Piece-to keys of the moves 1 and 2 plies before `ply` (-1 if none).
    int prev_move(int ply, int back) const {
        return ply >= back ? moved_[static_cast<std::size_t>(ply - back)] : -1;
    }

    // ThreatHist: squares the opponent attacks in the position at `ply`
    // (computed once per position, cached by its hash).
    std::uint64_t enemy_attacks(int ply) const {
        auto& c = threat_cache_[static_cast<std::size_t>(ply)];
        const std::uint64_t key = board_.hash();
        if (c.first == key) return c.second;
        using chess::PieceType;
        const chess::Color them = ~board_.sideToMove();
        const chess::Bitboard occ = board_.occ();
        std::uint64_t a = 0;
        auto each = [&](PieceType pt, auto&& f) {
            for (std::uint64_t b = board_.pieces(pt, them).getBits(); b; b &= b - 1)
                a |= f(chess::Square(static_cast<int>(std::countr_zero(b))));
        };
        each(PieceType::PAWN, [&](chess::Square sq) { return chess::attacks::pawn(them, sq).getBits(); });
        each(PieceType::KNIGHT, [&](chess::Square sq) { return chess::attacks::knight(sq).getBits(); });
        each(PieceType::BISHOP, [&](chess::Square sq) { return chess::attacks::bishop(sq, occ).getBits(); });
        each(PieceType::ROOK, [&](chess::Square sq) { return chess::attacks::rook(sq, occ).getBits(); });
        each(PieceType::QUEEN, [&](chess::Square sq) { return chess::attacks::queen(sq, occ).getBits(); });
        each(PieceType::KING, [&](chess::Square sq) { return chess::attacks::king(sq).getBits(); });
        c = {key, a};
        return a;
    }
    std::size_t threat_index(const chess::Move& m, int ply) const {
        const std::uint64_t att = enemy_attacks(ply);
        const std::size_t ft = (att >> m.from().index()) & 1, tt = (att >> m.to().index()) & 1;
        const std::size_t c = board_.sideToMove() == chess::Color::WHITE ? 0 : 1;
        return (((c * 4 + ft * 2 + tt) * 64 + static_cast<std::size_t>(m.from().index())) * 64) + static_cast<std::size_t>(m.to().index());
    }

    // Order score of a quiet move: butterfly history plus, if enabled, the
    // continuation history after the last two moves.
    int quiet_history(const chess::Move& m, int ply) const {
        int h = ThreatHist ? ot_.threat_hist[threat_index(m, ply)] : ot_.main.get(board_.sideToMove(), m);
        if (UseContHistory) {
            const int pt = piece_to(board_, m);
            h += ot_.cont_get(prev_move(ply, 1), pt, 1) + ot_.cont_get(prev_move(ply, 2), pt, 2);
            if (Cont4W > 0) h += ot_.cont_get(prev_move(ply, 4), pt, 4) * Cont4W / 64;
        }
        return h;
    }

    // Move-ordering score of a quiet move: quiet_history() plus the audit
    // additions (continuation history 3, 4 and 6 plies back, pawn history).
    // Only ordering uses these; LMR and pruning keep quiet_history().
    int order_history(const chess::Move& m, int ply) const {
        int h = quiet_history(m, ply);
        if (UseContHistory && (ContOrdW3 > 0 || ContOrdW4 > 0 || ContOrdW6 > 0)) {
            const int pt = piece_to(board_, m);
            h += (ot_.cont_get(prev_move(ply, 3), pt, 3) * ContOrdW3 + ot_.cont_get(prev_move(ply, 4), pt, 4) * ContOrdW4 +
                  ot_.cont_get(prev_move(ply, 6), pt, 6) * ContOrdW6) / 64;
        }
        if (PawnHistW > 0) h += ot_.pawn_hist[pawn_hist_index(m)] * PawnHistW / 64;
        if (LowPlyHist > 0 && ply < LowPlyN)
            h += low_ply_[static_cast<std::size_t>(ply) * 4096 + static_cast<std::size_t>(m.from().index() * 64 + m.to().index())] * LowPlyHist / 64;
        return h;
    }
    std::size_t pawn_hist_index(const chess::Move& m) const {
        const std::size_t k = mix_key(board_.pieces(chess::PieceType::PAWN, chess::Color::WHITE).getBits(),
                                      board_.pieces(chess::PieceType::PAWN, chess::Color::BLACK).getBits()) &
                              (OrderTables::PawnHistSize - 1);
        return k * 768 + static_cast<std::size_t>(piece_to(board_, m));
    }

    void update_quiet(const chess::Move& m, int ply, int bonus) {
        ot_.main.update(board_.sideToMove(), m, bonus);
        if (ThreatHist) update_stat(ot_.threat_hist[threat_index(m, ply)], bonus);
        if (UseContHistory) {
            const int pt = piece_to(board_, m);
            const bool use[7] = {false, true, true, ContOrdW3 > 0, Cont4W > 0 || ContOrdW4 > 0, false, ContOrdW6 > 0};
            for (int back = 1; back <= 6; ++back) {
                if (!use[back]) continue;
                const int prev = prev_move(ply, back);
                if (prev >= 0) update_stat(ot_.cont_at(prev, pt, back), bonus);
            }
        }
        if (PawnHistW > 0) update_stat(ot_.pawn_hist[pawn_hist_index(m)], bonus);
        if (LowPlyHist > 0 && ply < LowPlyN)
            update_stat(low_ply_[static_cast<std::size_t>(ply) * 4096 + static_cast<std::size_t>(m.from().index() * 64 + m.to().index())], bonus);
    }

    // Capture-history weight in capture ordering (/256): CaptHistOrdW when set.
    static int capt_hist_w() { return CaptHistOrdW > 0 ? CaptHistOrdW : CaptHistW; }
    // Capture history is kept up to date when any feature uses it.
    static bool capt_hist_on() { return capt_hist_w() > 0 || CaptHistSeeDiv > 0 || CapLmrOn > 0; }

    // Captures tracked by capture history (not promotions: those have their
    // own band and piece type changes).
    static bool is_capture_stat(const chess::Move& m) { return m.typeOf() != chess::Move::PROMOTION; }

    // Correction history keys for the current position (side to move included).
    struct CorrKeys {
        std::size_t pawn = 0, np_white = 0, np_black = 0;
        std::size_t minor = 0;
        std::size_t cont = SIZE_MAX;  // SIZE_MAX: no previous move (root, after a null move)
    };
    CorrKeys corr_keys(int ply) const {
        using chess::PieceType;
        using chess::Color;
        const std::size_t stm = board_.sideToMove() == Color::WHITE ? 0 : 1;
        const std::size_t mask = OrderTables::CorrSize - 1;
        auto np = [&](Color c) {
            return mix_key(board_.pieces(PieceType::KNIGHT, c).getBits() | board_.pieces(PieceType::BISHOP, c).getBits(),
                           board_.pieces(PieceType::ROOK, c).getBits() | board_.pieces(PieceType::QUEEN, c).getBits(),
                           board_.pieces(PieceType::KING, c).getBits());
        };
        CorrKeys k;
        k.pawn = stm * OrderTables::CorrSize +
                 (mix_key(board_.pieces(PieceType::PAWN, Color::WHITE).getBits(), board_.pieces(PieceType::PAWN, Color::BLACK).getBits()) & mask);
        k.np_white = (stm * 2 + 0) * OrderTables::CorrSize + (np(Color::WHITE) & mask);
        k.np_black = (stm * 2 + 1) * OrderTables::CorrSize + (np(Color::BLACK) & mask);
        if (CorrWMinor > 0) {
            auto minors = [&](Color c) {
                return board_.pieces(PieceType::KNIGHT, c).getBits() | board_.pieces(PieceType::BISHOP, c).getBits();
            };
            k.minor = stm * OrderTables::CorrSize +
                      (mix_key(minors(Color::WHITE), minors(Color::BLACK), board_.pieces(PieceType::KING).getBits()) & mask);
        }
        if (CorrWCont > 0) {
            const int prev = prev_move(ply, 1);
            if (prev >= 0) k.cont = stm * 768 + static_cast<std::size_t>(prev);
        }
        return k;
    }
    Score corrected(Score raw, const CorrKeys& k) const {
        const int c = (CorrWPawn * ot_.corr_pawn[k.pawn] + CorrWNonPawn * (ot_.corr_np[k.np_white] + ot_.corr_np[k.np_black]) +
                       (CorrWMinor > 0 ? CorrWMinor * ot_.corr_minor[k.minor] : 0) +
                       (CorrWCont > 0 && k.cont != SIZE_MAX ? CorrWCont * ot_.corr_cont[k.cont] : 0)) /
                      1024;
        return std::clamp(raw + c, -(Mate - MaxPly) + 1, Mate - MaxPly - 1);
    }
    void update_corr(const CorrKeys& k, int bonus) {
        auto upd = [&](std::int16_t& e) {
            const int v = e;
            e = static_cast<std::int16_t>(std::clamp(v + bonus - v * std::abs(bonus) / CorrLimit, -CorrLimit, CorrLimit));
        };
        upd(ot_.corr_pawn[k.pawn]);
        upd(ot_.corr_np[k.np_white]);
        upd(ot_.corr_np[k.np_black]);
        if (CorrWMinor > 0) upd(ot_.corr_minor[k.minor]);
        if (CorrWCont > 0 && k.cont != SIZE_MAX) upd(ot_.corr_cont[k.cont]);
    }

    // Make a move and remember its piece-to key for continuation history.
    void make(const chess::Move& m, int ply) {
        moved_[static_cast<std::size_t>(ply)] = piece_to(board_, m);
        played_[static_cast<std::size_t>(ply)] = m;
        played_quiet_[static_cast<std::size_t>(ply)] = is_quiet(board_, m);
        board_.make(m);
    }

    // PriorCmBonus: history bonus for the move made at ply - 1 (by the other
    // side), seen from the node at `ply`: main history of that side and the
    // continuation histories of that move.
    void prior_move_bonus(int ply, int bonus) {
        const chess::Move m = played_[static_cast<std::size_t>(ply - 1)];
        const int pt = moved_[static_cast<std::size_t>(ply - 1)];
        if (m == chess::Move::NO_MOVE || pt < 0) return;
        ot_.main.update(~board_.sideToMove(), m, bonus);
        if (UseContHistory) {
            const bool use[7] = {false, true, true, ContOrdW3 > 0, Cont4W > 0 || ContOrdW4 > 0, false, ContOrdW6 > 0};
            for (int back = 1; back <= 6; ++back) {
                if (!use[back]) continue;
                const int prev = prev_move(ply - 1, back);
                if (prev >= 0) update_stat(ot_.cont_at(prev, pt, back), bonus);
            }
        }
    }

    bool check_time() {
        if ((++nodes_ & 2047) == 0 && hooks_.should_stop(nodes_)) stopped_ = true;
        return stopped_;
    }

    // Repetition (RepRule), fifty-move rule, dead material.
    bool is_draw(int ply) const {
        if ((RepRule ? board_.repetition_draw(ply) : board_.isRepetition(1)) || board_.isInsufficientMaterial()) return true;
        if (!board_.isHalfMoveDraw()) return false;
        // Fifty-move rule: a checkmate on the 100th half-move still counts
        // (before the Oct 2026 audit: scored as a draw, so 7k/8/5K2/8/8/8/8/6Q1
        // w - - 99 missed Qg7#).
        if (board_.inCheck()) {
            chess::Movelist ml;
            chess::movegen::legalmoves(ml, board_);
            if (ml.empty()) return false;
        }
        return true;
    }

    bool root_move_allowed(const chess::Move& m) const {
        return std::find(root_moves_.begin(), root_moves_.end(), m) != root_moves_.end() &&
               std::find(excluded_.begin(), excluded_.end(), m) == excluded_.end();
    }

    // Triangular PV table: pv_[ply] holds the best line starting at ply.
    void update_pv(int ply, const chess::Move& move) {
        pv_[ply][ply] = move;
        for (int i = ply + 1; i < pv_len_[ply + 1]; ++i) pv_[ply][i] = pv_[ply + 1][i];
        pv_len_[ply] = std::max(pv_len_[ply + 1], ply + 1);
    }

    eval::EvalBoard board_;  // keeps its evaluation up to date as moves are made
    TranspositionTable& tt_;
    Hooks hooks_;

    std::vector<chess::Move> root_moves_;
    std::vector<chess::Move> excluded_;  // MultiPV: root moves already used
    chess::Move pv_move_ = chess::Move(chess::Move::NO_MOVE);  // best root move so far

    // Killer moves: two quiet moves per ply that recently caused cutoffs.
    std::array<std::array<chess::Move, 2>, MaxPly + 1> killers_{};
    std::unique_ptr<OrderTables> own_tables_;  // only when none was passed in
    OrderTables& ot_;
    // moved_[ply]: piece-to key of the move made at `ply` (-1 = null move).
    std::array<int, MaxPly + 1> moved_{};
    // played_[ply]: the move made at `ply`, and whether it was quiet (PriorCmBonus).
    std::array<chess::Move, MaxPly + 1> played_{};
    std::array<bool, MaxPly + 1> played_quiet_{};
    // LowPlyHist: [ply < 8][from * 64 + to], fresh for every search.
    std::vector<std::int16_t> low_ply_ = std::vector<std::int16_t>(8 * 4096);

    // Static eval per ply, for the "improving" flag (NoEval when in check).
    static constexpr Score NoEval = Infinite + 1;
    std::array<Score, MaxPly + 1> eval_stack_{};

    // Singular extensions: move excluded at each ply during a verification search.
    std::array<chess::Move, MaxPly + 1> se_excluded_{};
    std::array<int, MaxPly + 2> double_ext_{};  // double extensions on the path (SR_SE_DOUBLE)
    std::array<int, MaxPly + 2> cutoff_cnt_{};
    // ThreatHist: enemy attack maps per ply, keyed by the position's hash.
    mutable std::array<std::pair<std::uint64_t, std::uint64_t>, MaxPly + 2> threat_cache_{};
    // In check at each ply when the parent already knows (1 / 0), -1 = not known.
    std::array<std::int8_t, MaxPly + 2> known_check_ = [] { std::array<std::int8_t, MaxPly + 2> a{}; a.fill(-1); return a; }();
    // CheckExtSee: the move into this ply was a check that may be extended.
    std::array<bool, MaxPly + 2> ext_ok_ = [] { std::array<bool, MaxPly + 2> a{}; a.fill(true); return a; }();
    // RootOrder: nodes per root move in this iteration and the previous one.
    std::vector<std::pair<std::uint16_t, std::uint64_t>> cur_root_nodes_, prev_root_nodes_;
    int asp_vol_ = 0;  // AspAdaptive: average score change between iterations

    void add_root_nodes(const chess::Move& m, std::uint64_t n) {
        for (auto& [mv, cnt] : cur_root_nodes_)
            if (mv == m.move()) { cnt += n; return; }
        cur_root_nodes_.emplace_back(m.move(), n);
    }
    int root_order_score(const chess::Move& m) const {
        std::uint64_t mx = 1, mine = 0;
        for (const auto& [mv, cnt] : prev_root_nodes_) {
            mx = std::max(mx, cnt);
            if (mv == m.move()) mine = cnt;
        }
        return 1'100'000 + static_cast<int>(mine * 700'000 / mx);
    }  // beta cutoffs at each ply (reset per parent)
    int nmp_min_ply_ = 0;                        // null move off below this ply (verification)
    Score prev_score_ = 0;                     // last completed score (aspiration windows)
    bool have_prev_ = false;
    int root_depth_ = 1;
    int thread_id_ = 0;  // 0 = main thread

    std::array<std::array<chess::Move, MaxPly + 1>, MaxPly + 1> pv_{};
    std::array<int, MaxPly + 1> pv_len_{};

    std::uint64_t nodes_ = 0;
    int seldepth_ = 0;
    bool stopped_ = false;
};

// perft: counts leaf nodes to verify move generation against known values.
inline std::uint64_t perft(chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (depth <= 1) return static_cast<std::uint64_t>(moves.size());
    std::uint64_t total = 0;
    for (const auto& m : moves) {
        board.makeMove(m);
        total += perft(board, depth - 1);
        board.unmakeMove(m);
    }
    return total;
}

}  // namespace search

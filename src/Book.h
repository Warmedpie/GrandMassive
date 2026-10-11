// Book.h : opening book for GrandMassive.
//
// The lines in BookData.h are replayed once, on first use, into a table
// keyed by position hash (Zobrist), so transpositions into a book position
// are found too. A move's weight is the number of book lines that play it
// from that position; its score is the cached depth-10 search score (White's
// view) of the position it leads to, computed by tools/make_book.
//
// Book modes (feature flag BOOK_MODE sets the default; UCI option "BookMode"
// changes it at run time). The modes form a scale, widest to narrowest:
// Each mode's cp value is a one-sided loss limit, a hard rule: the book never
// plays a move whose resulting score, from the mover's side, is below -limit.
// Moves that leave us better are always allowed, however large the edge, so if
// the opponent strays into a book line that is bad for them we take the
// punishing reply (the book only stores lines to +/-200, the build-time cut).
//   Wide    (0) - weighted-random among moves that lose at most 200 cp,
//                 and also plays WideLines: deliberately imbalanced openings
//                 (gambits, offbeat lines) that leave the book at 100..200 cp,
//                 weighted BOOK_WIDE_WEIGHT each
//   Normal  (1) - same, losing at most 100 cp, book lines only (the default)
//   Tight   (2) - same, losing at most 50 cp
//   Optimal (3) - the move with the best cached score for the side to move
//                 (random among moves within BOOK_OPT_MARGIN = 10 cp of it,
//                 i.e. depth-10 noise; 29 distinct lines vs 1 at 5 cp), so the
//                 "reduced search" is paid once, at book build time.
// A mode's filter can leave no moves in a book position; the engine then
// leaves the book and searches, the same as out of book.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "BookData.h"
#include "Chess.h"

#ifndef BOOK_MODE
#define BOOK_MODE 1  // 0 Wide, 1 Normal, 2 Tight, 3 Optimal
#endif

namespace book {

enum class Mode { Wide = 0, Normal = 1, Tight = 2, Optimal = 3 };

inline constexpr const char* ModeNames[] = {"Wide", "Normal", "Tight", "Optimal"};
#ifndef BOOK_OPT_MARGIN
#define BOOK_OPT_MARGIN 10
#endif
inline constexpr int OptimalMargin = BOOK_OPT_MARGIN;  // cp: ties for Optimal, keeps a little variety

#ifndef BOOK_WIDE_WEIGHT
#define BOOK_WIDE_WEIGHT 2
#endif
inline constexpr int WideWeight = BOOK_WIDE_WEIGHT;  // weight of one WideLines line vs a normal line

// Loss limit (cp, mover's view): the book never plays into a score below -limit.
inline constexpr int mode_bound(Mode m) {
    switch (m) {
        case Mode::Wide: return 200;
        case Mode::Tight: return 50;
        default: return 100;  // Normal; Optimal picks among the Normal set
    }
}

struct BookMove {
    chess::Move move;
    int weight = 0;       // number of normal book lines playing this move here
    int wide_weight = 0;  // number of WideLines playing it (Wide mode only)
    int score = 0;  // cp, White's view, of the position after the move
};

class Book {
public:
    static Book& instance() {
        static Book book;
        return book;
    }

    Mode mode() const { return mode_; }
    void set_mode(Mode m) { mode_ = m; }

    // Legal book moves for this position, most-played first, unfiltered by
    // mode. Empty if the position isn't in the book.
    std::vector<BookMove> moves(const chess::Board& board) const {
        std::vector<BookMove> result;
        const auto it = table_.find(board.hash());
        if (it == table_.end()) return result;

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        for (const auto& entry : it->second)
            for (const auto& m : legal)
                if (m.move() == entry.move.move()) {  // guards against hash collisions
                    result.push_back(entry);
                    break;
                }
        std::sort(result.begin(), result.end(),
                  [](const BookMove& a, const BookMove& b) {
                      return a.weight != b.weight ? a.weight > b.weight : a.wide_weight > b.wide_weight;
                  });
        return result;
    }

    // Book moves this mode allows here; weight is the mode's effective weight.
    std::vector<BookMove> playable(const chess::Board& board, Mode m) const {
        auto list = moves(board);
        const int limit = mode_bound(m);
        const int sign = board.sideToMove() == chess::Color::WHITE ? 1 : -1;  // White's view -> mover's
        if (m == Mode::Wide)
            for (auto& c : list) c.weight += WideWeight * c.wide_weight;
        list.erase(std::remove_if(list.begin(), list.end(),
                                  [&](const BookMove& c) { return sign * c.score < -limit || c.weight == 0; }),
                   list.end());
        if (m == Mode::Optimal && !list.empty()) {
            int best = sign * list.front().score;
            for (const auto& c : list) best = std::max(best, sign * c.score);
            list.erase(std::remove_if(list.begin(), list.end(),
                                      [&](const BookMove& c) { return sign * c.score < best - OptimalMargin; }),
                       list.end());
        }
        return list;
    }

    // A book move for the current mode (weighted-random), or NO_MOVE.
    chess::Move probe(const chess::Board& board) { return probe(board, mode_); }

    chess::Move probe(const chess::Board& board, Mode m) {
        const auto candidates = playable(board, m);
        int total = 0;
        for (const auto& c : candidates) total += c.weight;
        if (total == 0) return chess::Move(chess::Move::NO_MOVE);

        int r = std::uniform_int_distribution<int>(0, total - 1)(rng_);
        for (const auto& c : candidates) {
            if (r < c.weight) return c.move;
            r -= c.weight;
        }
        return candidates.front().move;
    }

    std::size_t positions() const { return table_.size(); }

private:
    Book() : mode_(static_cast<Mode>(BOOK_MODE)), rng_(std::random_device{}()) {
        for (const std::string_view line : BookLines) load(line, false);
        for (const std::string_view line : WideLines) load(line, true);
    }

    void load(std::string_view line, bool wide) {
        chess::Board board;
        std::istringstream is{std::string(line)};
        for (std::string tok; is >> tok;) {
            const auto at = tok.find('@');  // "e2e4@+31"
            const int score = at == std::string::npos ? 0 : std::atoi(tok.c_str() + at + 1);
            const chess::Move m = chess::uci::uciToMove(board, tok.substr(0, at));
            if (m == chess::Move::NO_MOVE) break;  // lines are pre-validated; be safe anyway
            add(board.hash(), m, score, wide);
            board.makeMove(m);
        }
    }

    void add(std::uint64_t key, const chess::Move& m, int score, bool wide) {
        auto& list = table_[key];
        for (auto& entry : list)
            if (entry.move.move() == m.move()) {
                ++(wide ? entry.wide_weight : entry.weight);
                return;
            }
        list.push_back({m, wide ? 0 : 1, wide ? 1 : 0, score});
    }

    std::unordered_map<std::uint64_t, std::vector<BookMove>> table_;
    Mode mode_;
    std::mt19937 rng_;
};

}  // namespace book

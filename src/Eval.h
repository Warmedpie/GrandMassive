// Eval.h : Position evaluation for GrandMassive.
//
// Tapered evaluation:
//   eval = taper(mg, eg, phase), mg / eg = white's terms - black's terms,
// terms = material + piece-square tables (PST) + mobility, phase from the
// non-pawn material on the board. Then scaled down as the fifty-move rule
// approaches.
//
// Piece values (PeSTO) and mobility weights come from the author's earlier
// engine; A/B tested at fixed depth 4 (2000 games each): PeSTO values +26 Elo,
// mobility +76 Elo. The earlier engine's per-side phase tested -20 / -31 Elo
// against this global phase and was not kept.
//
// Incremental: EvalBoard overrides chess::Board's placePiece / removePiece
// hooks, which makeMove / unmakeMove call for every piece that appears or
// disappears; each call adds / subtracts table entries. Everything that
// depends only on (piece, square) lives in those tables, including all of
// knight mobility and the constant part of slider mobility. evaluate() only
// has to add slider mobility (depends on occupancy) and do the phase blend.

#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <bit>
#include <cstdlib>
#include <cstdint>
#include <string_view>

#include "Chess.h"
#include "Endgames.h"

#ifndef SP_LAZY_ACC
#define SP_LAZY_ACC 1  // lazy accumulator updates in the search (nnue::AccStack)
#endif
#ifndef EV_NNUE
#define EV_NNUE 1  // 1: NNUE evaluation (Nnue.h, NetData.h) in the search; 0: hand-written eval
#endif
#if EV_NNUE
#include "Nnue.h"
#endif

namespace eval {

// Scores are in centipawns-like units (a middlegame pawn = 82), used as UCI
// "score cp".
using Score = int;

// ---------------------------------------------------------------------------
// Piece values (PeSTO), indexed by chess::PieceType (P, N, B, R, Q, K).
// ---------------------------------------------------------------------------
inline constexpr std::array<Score, 6> PieceValueMG = {82, 337, 365, 497, 1025, 0};
inline constexpr std::array<Score, 6> PieceValueEG = {94, 281, 297, 512, 936, 0};

// Named middlegame values (used by SEE, MVV-LVA and pruning margins).
inline constexpr Score PawnValue   = PieceValueMG[0];
inline constexpr Score KnightValue = PieceValueMG[1];
inline constexpr Score BishopValue = PieceValueMG[2];
inline constexpr Score RookValue   = PieceValueMG[3];
inline constexpr Score QueenValue  = PieceValueMG[4];

// Search bounds. Mate scores sit just below Infinite so the search can
// encode "mate in N" as Mate - ply and still stay inside the window.
inline constexpr Score Infinite = 32000;
inline constexpr Score Mate     = 31000;

// ---------------------------------------------------------------------------
// Game phase (Stockfish's scheme, scaled to our piece values).
//
// Non-pawn material (knights, bishops, rooks, queens at middlegame values,
// both sides) at the start is 2 * (2*337 + 2*365 + 2*497 + 1025) = 6846.
// Stockfish's limits sit at 91.9% and 23.6% of its starting value (15258 and
// 3915 of 16604), which scale to 6291 and 1614 here. Above MidgameLimit is
// full middlegame (phase 128), below EndgameLimit full endgame (0).
// ---------------------------------------------------------------------------
inline constexpr std::array<Score, 6> PhaseWeight = {
    0, PieceValueMG[1], PieceValueMG[2], PieceValueMG[3], PieceValueMG[4], 0,
};
inline constexpr Score MidgameLimit = 6291;
inline constexpr Score EndgameLimit = 1614;
inline constexpr int PhaseMax = 128;

inline constexpr int phase_from_npm(Score npm) {
    npm = std::clamp(npm, EndgameLimit, MidgameLimit);
    return (npm - EndgameLimit) * PhaseMax / (MidgameLimit - EndgameLimit);
}

inline constexpr Score taper(Score mg, Score eg, int phase) {
    return (mg * phase + eg * (PhaseMax - phase)) / PhaseMax;
}

// ---------------------------------------------------------------------------
// Mobility: bonus = weight * (attacked squares - base) per piece. With the
// mobility-area term on (default), squares holding our own pawns or attacked
// by enemy pawns don't count. Started from the old engine's weights; tuned.
// ---------------------------------------------------------------------------
struct MobilityWeight {
    Score mg, eg;
    int base;
};
inline constexpr MobilityWeight KnightMobility = {8, 11, 4};
inline constexpr MobilityWeight BishopMobility = {5, -1, 7};
inline constexpr MobilityWeight RookMobility = {3, 9, 7};
inline constexpr MobilityWeight QueenMobility = {6, -3, 14};

// ---------------------------------------------------------------------------
// Piece-square tables, from WHITE's point of view.
//
// Laid out the way the board looks from White's side: the first row is
// rank 8 (a8 ... h8), the last row is rank 1 (a1 ... h1). Black uses the
// same tables mirrored vertically, so each table is written once.
//
// Values are Texel-tuned (tools/tuner.cpp) in engine units, starting from
// Stockfish's classical PSQT scaled to our piece values. Squares too rare in
// the tuning data to tune (e.g. a knight on the 8th rank) kept their
// starting values, which is why those rows still look symmetric.
// ---------------------------------------------------------------------------
using Table = std::array<Score, 64>;

// PST scale. Before tuning, the tables held raw Stockfish values scaled by
// our starting non-pawn material / Stockfish's, per side & phase:
//   opening: 2*337 + 2*365 + 2*497 + 1025 = 3423  vs  2*781 + 2*825 + 2*1276 + 2538 = 8302
//   endgame: 2*281 + 2*297 + 2*512 +  936 = 3116  vs  2*854 + 2*915 + 2*1380 + 2682 = 8980
// Since Texel tuning the tables are in engine units, so the scale is 1:1.
inline constexpr Score PstScaleNumMG = 1;
inline constexpr Score PstScaleDenMG = 1;
inline constexpr Score PstScaleNumEG = 1;
inline constexpr Score PstScaleDenEG = 1;

// clang-format off
// ---- Opening / middlegame --------------------------------------------------
// Pawns, opening / middlegame
inline constexpr Table PawnMG = {
       0,   0,   0,   0,   0,   0,   0,   0,   // rank 8
      -3,   3,  -1,  -5,   2,  -7,   4,  -3,   // rank 7
     -23, -27, -17, -32, -25, -19, -48, -13,   // rank 6
     -32, -35, -34, -42, -35, -33, -43, -31,   // rank 5
     -38, -48, -30, -29, -30, -28, -49, -47,   // rank 4
     -34, -33, -26, -33, -21, -19, -13, -36,   // rank 3
     -38, -24, -25, -32, -28, -14,  -7, -44,   // rank 2
       0,   0,   0,   0,   0,   0,   0,   0,   // rank 1
};

// Knights, opening / middlegame
inline constexpr Table KnightMG = {
     -83, -34, -23, -11, -11, -23, -34, -83,   // rank 8
     -28, -11,  11,  15,  15,   2, -11, -28,   // rank 7
      -4,  22,  16,  33,  36,  25,  34,  -4,   // rank 6
       4,  18,  23,  40,  28,  35,  21,  24,   // rank 5
      10,  12,  22,  12,  16,  32,  14,  13,   // rank 4
     -16, -10,  -9,   3,  -1,  -6,  -1, -19,   // rank 3
     -32, -17, -21, -14, -14, -20,  -1, -15,   // rank 2
     -72, -15, -31, -16,  -1, -28, -18, -72,   // rank 1
};

// Bishops, opening / middlegame
inline constexpr Table BishopMG = {
     -20,   0,  -6,  -9,  -9,  -6,   0, -20,   // rank 8
      -7, -12, -13,   0,   0,   2,  -6,  -7,   // rank 7
      -2,  14,  -1,  24,  26, -13,  -3,   5,   // rank 6
       4,  17,  19,  47,  21,  16,  16,  -4,   // rank 5
       8,   8,  17,  17,  25,  12,  -6,   6,   // rank 4
      18,  23,  -9,   3,  -2,  -2,   7,   1,   // rank 3
      -2,   5,  10, -14,  -8,   2,   1,  -6,   // rank 2
     -22,  -2, -10, -10,   0, -22,  -2, -11,   // rank 1
};

// Rooks, opening / middlegame
inline constexpr Table RookMG = {
      -7,  -8,   0,   4,   4,   0,  -8,  -7,   // rank 8
       3, -13,   5,  31,   0,   7,   5,  -8,   // rank 7
     -40, -28,  -4,   1,  -9,   2,  -1,  -9,   // rank 6
     -36,  -8, -29, -24, -26, -14, -32, -25,   // rank 5
     -35, -29, -34, -36, -15,  -8,  -9, -27,   // rank 4
     -38, -30, -35, -27, -33, -14, -29, -39,   // rank 3
     -36, -20, -24, -26, -21, -14, -19, -52,   // rank 2
     -31, -27, -23, -17,  -8, -11, -28, -20,   // rank 1
};

// Queens, opening / middlegame
inline constexpr Table QueenMG = {
      -1,  -1,   0,  -1,  -1,   0,  -1,  -1,   // rank 8
     -51, -23,  -7,   3,   3,  -6,   2,  12,   // rank 7
      -3,   1,  -6,  14,  10,  10, -10,   1,   // rank 6
     -15,  -9, -15, -24, -31,   0, -17,   1,   // rank 5
      -2, -21, -11, -19,  -9,  -8,  -1, -17,   // rank 4
     -15,  -2, -19, -14, -13,  -8,   1, -13,   // rank 3
     -22, -15,  -8,  -8, -10,   3,   7,  -1,   // rank 2
     -13, -21, -16, -15, -11, -31,  -2,   1,   // rank 1
};

// King, opening / middlegame
inline constexpr Table KingMG = {
      24,  37,  19,   0,   0,  19,  37,  24,   // rank 8
      36,  49,  27,  14,  14,  27,  49,  36,   // rank 7
      51,  60,  33,  13,  13,  33,  60,  51,   // rank 6
      63,  74,  43,  29,  29,  43,  74,  63,   // rank 5
      68,  78,  57,  40,  40,  57,  78,  68,   // rank 4
      80,  86,  29,  27,  33,  62,  96,  96,   // rank 3
     115, 165, 141, 139, 117, 159, 198, 198,   // rank 2
     112, 212, 197, 162, 167, 169, 215, 217,   // rank 1
};

// ---- Endgame ---------------------------------------------------------------
// Pawns, endgame
inline constexpr Table PawnEG = {
       0,   0,   0,   0,   0,   0,   0,   0,   // rank 8
     -27,  -8,  15,  10,  -7,   7, -15, -29,   // rank 7
      22,  24,  20,  24,  27,  25,  36,  22,   // rank 6
      15,  17,  11,  14,  12,  16,  18,  11,   // rank 5
       7,  13,   8,   9,  11,   8,   9,   4,   // rank 4
       2,   4,  11,  10,  14,  14,   2,   4,   // rank 3
       5,   1,   8,  21,  18,  18,   5,   1,   // rank 2
       0,   0,   0,   0,   0,   0,   0,   0,   // rank 1
};

// Knights, endgame
inline constexpr Table KnightEG = {
     -35, -31, -23,   9,  -6, -19, -31, -35,   // rank 8
     -19,  -7,   3,   6,   8,  -4,  -8, -12,   // rank 7
      -2,  -6,   0,   7,   3,   0,  -2, -17,   // rank 6
      -3,  -1,   1,  12,  12,   4,  -2, -15,   // rank 5
      -2,   0,   3,   8,  11,   1,   3,  -4,   // rank 4
     -32,  -7, -10, -11,  -2, -17, -19, -31,   // rank 3
     -23, -11, -24,  -6, -19, -18, -31, -19,   // rank 2
     -33, -34, -21,  -7, -21, -25, -32, -33,   // rank 1
};

// Bishops, endgame
inline constexpr Table BishopEG = {
      -8,   7,   4,   8,  10,   9, -15, -16,   // rank 8
       1,  -3,  14,  11,   6,   7,   5,  -8,   // rank 7
      10,  24,   2,  20,  11,  19,  14,   0,   // rank 6
       7,  16,  15,  20,  25,  20,  13,   1,   // rank 5
      18,  14,  15,  24,  18,  20,   4,   3,   // rank 4
      -5,  15,   3,  10,  16, -14,   6,   0,   // rank 3
      -1, -14,  -3,   7,  -1,  -2, -18,   0,   // rank 2
       0,  -7,  -3,  -1,  -1,  -1, -11, -23,   // rank 1
};

// Rooks, endgame
inline constexpr Table RookEG = {
      23,  24,  22,  23,  19,  28,  28,  23,   // rank 8
      17,  16,  15,  15,   8,  18,  16,  17,   // rank 7
      26,  22,  15,  16,  15,  20,  19,  12,   // rank 6
      22,  21,  18,  14,   8,  10,  13,  10,   // rank 5
      16,  14,  14,  11,  -1,   1,   5,   0,   // rank 4
       5,   5,   1,  -1,  -4,  -2,  -8,   1,   // rank 3
      -1,  -9,  -7,  -9, -11, -13, -19, -18,   // rank 2
      -8, -13,  -8, -11, -18, -16, -14, -21,   // rank 1
};

// Queens, endgame
inline constexpr Table QueenEG = {
      69,  59,  78,  99,  87,  84, 103,  86,   // rank 8
      85,  80,  84,  87,  83,  73,  67,  63,   // rank 7
      52,  88,  75,  93,  89,  81,  93,  58,   // rank 6
      53,  66,  86, 101,  92,  80,  75,  52,   // rank 5
      30,  75,  56,  76,  64,  60,  52,  34,   // rank 4
      40,  36,  64,  50,  48,  52,  29,  51,   // rank 3
     -19,  33,  18,  24,  26,   0, -12, -19,   // rank 2
     -24, -20,  18,  10,  -2, -21, -20, -24,   // rank 1
};

// King, endgame
inline constexpr Table KingEG = {
       4,  49,  38,  35,  64,  48,  48,   4,   // rank 8
      39,  88,  89,  84,  87,  88,  85,  35,   // rank 7
      90, 105, 105, 106, 110, 108, 112,  77,   // rank 6
      74,  90,  98, 102,  98,  99,  91,  64,   // rank 5
      54,  72,  84,  87,  83,  84,  70,  62,   // rank 4
      46,  59,  71,  75,  76,  71,  64,  50,   // rank 3
      53,  58,  64,  63,  66,  61,  56,  46,   // rank 2
      40,  39,  42,  31,  35,  44,  36,  20,   // rank 1
};
// clang-format on

// Indexed by chess::PieceType.
inline constexpr std::array<const Table*, 6> PSTMG = {
    &PawnMG, &KnightMG, &BishopMG, &RookMG, &QueenMG, &KingMG,
};
inline constexpr std::array<const Table*, 6> PSTEG = {
    &PawnEG, &KnightEG, &BishopEG, &RookEG, &QueenEG, &KingEG,
};

// ---------------------------------------------------------------------------
// Lookup tables built at compile time. Index: [piece][square], piece =
// color * 6 + type (chess::Piece order), square = a1 = 0 ... h8 = 63.
// Values are from the owner's point of view (positive = good for the side
// the piece belongs to); they are summed per side.
// ---------------------------------------------------------------------------
namespace detail {

// Table row for chess square `sq` (a1 = 0) as seen by the piece's owner.
// Tables are written rank 8 first, so White flips the rank (sq ^ 56) and
// Black, whose rank 8 is White's rank 1, reads the square directly.
constexpr std::size_t table_index(int sq, bool white) {
    return static_cast<std::size_t>(white ? (sq ^ 56) : sq);
}

// v * num / den, rounded to nearest (half away from zero).
constexpr Score scale(Score v, Score num, Score den) {
    const long long x = static_cast<long long>(v) * num;
    return static_cast<Score>((x >= 0 ? x + den / 2 : x - den / 2) / den);
}

// Knight moves from a square on an empty board (knight attacks don't
// depend on occupancy, so knight mobility is a pure piece-square term).
// (Written with plain index loops and no lambdas so older MSVC versions can
// evaluate it at compile time.)
constexpr int knight_squares(int sq) {
    const int df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    const int dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    int n = 0;
    for (int i = 0; i < 8; ++i) {
        const int f = sq % 8 + df[i], r = sq / 8 + dr[i];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) ++n;
    }
    return n;
}

// Mobility, incremental part for one piece on one square: all of it for
// knights, and the constant "- weight * base" for sliders (their square
// counts depend on occupancy and are added in evaluate()).
constexpr Score mobility_constant(int type, int sq, bool mg) {
    switch (type) {
        case 1: return (mg ? KnightMobility.mg : KnightMobility.eg) * (knight_squares(sq) - KnightMobility.base);
        case 2: return -(mg ? BishopMobility.mg : BishopMobility.eg) * BishopMobility.base;
        case 3: return -(mg ? RookMobility.mg : RookMobility.eg) * RookMobility.base;
        case 4: return -(mg ? QueenMobility.mg : QueenMobility.eg) * QueenMobility.base;
        default: return 0;
    }
}

using PieceSquare = std::array<std::array<Score, 64>, 12>;

enum class Kind { MaterialMG, MaterialEG, PstMG, PstEG, MobilityMG, MobilityEG, NonPawnMaterial };

constexpr PieceSquare build(Kind kind) {
    PieceSquare out{};
    for (int color = 0; color < 2; ++color) {
        const bool white = (color == 0);
        for (int type = 0; type < 6; ++type) {
            const auto t = static_cast<std::size_t>(type);
            for (int sq = 0; sq < 64; ++sq) {
                Score v = 0;
                switch (kind) {
                    case Kind::MaterialMG: v = PieceValueMG[t]; break;
                    case Kind::MaterialEG: v = PieceValueEG[t]; break;
                    case Kind::PstMG: v = scale((*PSTMG[t])[table_index(sq, white)], PstScaleNumMG, PstScaleDenMG); break;
                    case Kind::PstEG: v = scale((*PSTEG[t])[table_index(sq, white)], PstScaleNumEG, PstScaleDenEG); break;
                    case Kind::MobilityMG: v = mobility_constant(type, sq, true); break;
                    case Kind::MobilityEG: v = mobility_constant(type, sq, false); break;
                    case Kind::NonPawnMaterial: v = PhaseWeight[t]; break;
                }
                out[static_cast<std::size_t>(color * 6 + type)][static_cast<std::size_t>(sq)] = v;
            }
        }
    }
    return out;
}

inline constexpr PieceSquare MaterialMGTable = build(Kind::MaterialMG);
inline constexpr PieceSquare MaterialEGTable = build(Kind::MaterialEG);
inline constexpr PieceSquare PstMGTable      = build(Kind::PstMG);
inline constexpr PieceSquare PstEGTable      = build(Kind::PstEG);
inline constexpr PieceSquare MobilityMGTable = build(Kind::MobilityMG);
inline constexpr PieceSquare MobilityEGTable = build(Kind::MobilityEG);
inline constexpr PieceSquare NPMTable        = build(Kind::NonPawnMaterial);

constexpr std::size_t pi(chess::Piece p) { return static_cast<std::size_t>(p.internal()); }
constexpr std::size_t si(chess::Square sq) { return static_cast<std::size_t>(sq.index()); }

}  // namespace detail

// Owner's-view table lookups for one piece on one square.
inline Score pst_mg_of(chess::Piece p, chess::Square sq) { return detail::PstMGTable[detail::pi(p)][detail::si(sq)]; }
inline Score pst_eg_of(chess::Piece p, chess::Square sq) { return detail::PstEGTable[detail::pi(p)][detail::si(sq)]; }

// ---------------------------------------------------------------------------
// Incrementally maintained terms, one set per side, each from that side's
// own point of view (all positive = good for that side).
// ---------------------------------------------------------------------------
struct SideTerms {
    Score material_mg = 0, material_eg = 0;
    Score pst_mg = 0, pst_eg = 0;
    Score mobility_mg = 0, mobility_eg = 0;  // incremental part only
    Score npm = 0;                           // non-pawn material (PhaseWeight): drives the phase

    bool operator==(const SideTerms&) const = default;
};

struct Terms {
    std::array<SideTerms, 2> side;  // [0] = white, [1] = black

    int phase() const { return phase_from_npm(side[0].npm + side[1].npm); }

    void add(chess::Piece p, chess::Square sq) { apply(p, sq, +1); }
    void sub(chess::Piece p, chess::Square sq) { apply(p, sq, -1); }

    bool operator==(const Terms&) const = default;

private:
    void apply(chess::Piece p, chess::Square sq, int sign) {
        using namespace detail;
        SideTerms& s = side[static_cast<std::size_t>(p.color().internal())];
        const std::size_t i = pi(p), j = si(sq);
        s.material_mg += sign * MaterialMGTable[i][j];
        s.material_eg += sign * MaterialEGTable[i][j];
        s.pst_mg      += sign * PstMGTable[i][j];
        s.pst_eg      += sign * PstEGTable[i][j];
        s.mobility_mg += sign * MobilityMGTable[i][j];
        s.mobility_eg += sign * MobilityEGTable[i][j];
        s.npm         += sign * NPMTable[i][j];
    }
};

// From-scratch computation: walks each piece bitboard, popping one set bit
// per piece. Used to initialize EvalBoard, for the UCI "eval" command, and
// to cross-check the incremental terms.
inline Terms compute(const chess::Board& board) {
    Terms t;
    for (int color = 0; color < 2; ++color) {
        const chess::Color c = color == 0 ? chess::Color::WHITE : chess::Color::BLACK;
        for (int type = 0; type < 6; ++type) {
            const chess::PieceType pt(static_cast<chess::PieceType::underlying>(type));
            const chess::Piece piece(pt, c);
            chess::Bitboard bb = board.pieces(pt, c);
            while (bb) t.add(piece, chess::Square(bb.pop()));
        }
    }
    return t;
}

// ---------------------------------------------------------------------------
// Slider mobility, occupancy-dependent part: weight * attacked squares,
// summed per piece type (the "- weight * base" part is incremental).
// ---------------------------------------------------------------------------
struct MgEg {
    Score mg = 0, eg = 0;
};

inline MgEg slider_mobility(const chess::Board& b, chess::Color c) {
    const chess::Bitboard occ = b.occ();
    int bishop = 0, rook = 0, queen = 0;

    chess::Bitboard bb = b.pieces(chess::PieceType::BISHOP, c);
    while (bb) bishop += chess::attacks::bishop(chess::Square(bb.pop()), occ).count();
    bb = b.pieces(chess::PieceType::ROOK, c);
    while (bb) rook += chess::attacks::rook(chess::Square(bb.pop()), occ).count();
    bb = b.pieces(chess::PieceType::QUEEN, c);
    while (bb) queen += chess::attacks::queen(chess::Square(bb.pop()), occ).count();

    return {BishopMobility.mg * bishop + RookMobility.mg * rook + QueenMobility.mg * queen,
            BishopMobility.eg * bishop + RookMobility.eg * rook + QueenMobility.eg * queen};
}

// ===========================================================================
// Positional terms (computed per node, not incrementally).
//
// Each term has an on/off switch and named weights (mg, eg), in the same
// units as the piece values. Weights are Texel-tuned (tools/tuner.cpp, 612k
// self-play positions; +42 Elo at 50 ms/move over the hand-set values),
// except the king-danger table and attack units, which the tuner holds fixed.
// The EV_* macros exist only so A/B test builds can flip terms from the
// compiler command line; normal builds use the defaults below.
//
// A/B results (fixed depth 4, each term added on top of the ones before it):
//   passed pawns      +68  (2000 games)      kept
//   king safety       +15  (2000)            kept
//   threats           +59  (2000)            kept
//   bishop pair / bad +32  (2000)            kept
//   rook terms        +11  (2000, LOS 94%)   kept
//   mobility area     +28  (2000)            kept
//   tempo             +23  (2000)            kept
//   pawn structure    +25  (3000; +4 on an earlier, weaker baseline)  kept
//   space             +16  (3000)            kept
//   outposts           +5  (3000 twice, LOS ~82%)  off: not proven, retune
//   endgame scaling    -2  (3000)            off
// ===========================================================================
#ifndef EV_PASSED
#define EV_PASSED 1
#endif
#ifndef EV_KING_SAFETY
#define EV_KING_SAFETY 1
#endif
#ifndef EV_PAWN_STRUCTURE
#define EV_PAWN_STRUCTURE 1
#endif
#ifndef EV_OUTPOSTS
#define EV_OUTPOSTS 0
#endif
#ifndef EV_BISHOPS
#define EV_BISHOPS 1
#endif
#ifndef EV_ROOKS
#define EV_ROOKS 1
#endif
#ifndef EV_THREATS
#define EV_THREATS 1
#endif
#ifndef EV_MOBILITY_AREA
#define EV_MOBILITY_AREA 1
#endif
#ifndef EV_TEMPO
#define EV_TEMPO 1
#endif
#ifndef EV_SPACE
#define EV_SPACE 1
#endif
// Endgame terms, added after a loss to Stockfish where the engine misjudged
// a rook + minor endgame (outside passed pawn, bad bishop, king invasion).
// A/B tests, 25 ms/move vs. the engine without them; "EG suite" = 800
// balanced endgame start positions (tools/eg_suite.epd), "openings" = the
// usual random 8-ply openings:
//   all six, hand-set values        EG suite  +9 (600)   openings -23 (600, LOS 3%)
//   each alone on the EG suite: passed extra -2, bad bishop x blocked -5,
//     king activity +2 (600 each); pawn threats, rook vs passer ~0 (partial)
//   Texel-tuned, bad bishop x blocked dropped (tuned to {-1,-1}):
//     five terms                    EG suite +14 (600, LOS 96%)   openings -41 (600)
//     four, without pawn threats    EG suite -16 (300)            openings -33 (550)
//     pawn threats, eg weight only  EG suite ~0  (750)            openings +4  (500)
// None is a proven gain at this time control, and together they cost
// strength from normal openings, so all stay off.
//
// Then retuned on 240k Stockfish 19 labelled positions (tools/data, expected
// score from SF's WDL at 20k nodes), EG suite re-balanced by Stockfish (566):
//   v2: all eg weights (PSTs, passers, structure, ...), mg fixed, terms off
//       vs. before:  EG suite +26 (600, LOS 100%)   openings +27 (600, LOS 99%)   KEPT
//   v3: same plus these terms on and tuned
//       vs. before:  EG suite +24 (600)              openings +10 (600)
//       vs. v2:      EG suite +13 (800, LOS 98%)    openings -11 (800) and behind in a 2nd run
//   v1: only these terms tuned: EG suite +11 (600, LOS 95%), openings -10 (600)
// So the terms help endgames a little but still cost more elsewhere: off.
//
// Then all weights, mg included (tuner mode 0, l2 0), on the same labels with
// the endgame part re-scored at 100k nodes:
//   vs. the eg-only tune:  openings +38 (600, LOS 99.9%), confirmed +38 in a
//                          2nd run with new openings; EG suite +8 and -2 (600 each)   KEPT
//   (with a light L2 penalty, 1e-8: openings ~0, not kept)
//
// Second round, from comparing with Stockfish 11's hand-written eval on the
// Stockfish game (its threats / bishops / imbalance terms explained most of
// the gap). All weights re-tuned together (mode 0) on the Stockfish labels:
//   A: pawn threats (on again) + threats on pawns no pawn defends + bad bishop
//      per fixed pawn + knight / rook value by own pawn count
//      vs. before:  openings +23 and +17 (600 each; combined +20, LOS 99.3%)
//                   EG suite +13 (600, LOS 96%)                                  KEPT
//   B: the same without the older pawn-threat term: openings -8 (600)
//
// Third round (removed): king activity (near enemy weak pawns / own pawns),
// king close to the pawn centroid, bad bishop scaled by blocked centre pawns,
// and an "initiative" term pushing the endgame score toward the side ahead.
// They fit the Stockfish labels 2% better but lost in games, tuned on labels
// (openings -28 / -45 / -23 / -12 for various subsets) and tuned by SPSA on
// 10,000 games (openings -33, EG suite -2). tools/symcheck found and fixed a
// colour asymmetry on the way; it wasn't the cause. Code removed.
#ifndef EV_PASSED_EXTRA
#define EV_PASSED_EXTRA 0
#endif
#ifndef EV_CANDIDATE
#define EV_CANDIDATE 0
#endif
#ifndef EV_PAWN_THREATS
#define EV_PAWN_THREATS 1
#endif
#ifndef EV_ROOK_PASSERS
#define EV_ROOK_PASSERS 0
#endif
// Second round, guided by where Stockfish 11's hand-written eval disagreed
// with ours in the Stockfish game (see the results block further down).
#ifndef EV_WEAK_PAWN_THREATS
#define EV_WEAK_PAWN_THREATS 1
#endif
#ifndef EV_BAD_BISHOP3
#define EV_BAD_BISHOP3 1
#endif
#ifndef EV_IMBALANCE
#define EV_IMBALANCE 1
#endif
// King danger rules (see danger_penalty below).
//   EV_KS_ONE_ATTACKER: 0 = penalty needs 2+ pieces hitting the king zone;
//     1 = one is enough when the attacker has a queen; 2 = one is enough when
//     the attacker's queen itself hits the king zone (line of sight).
//   EV_KS_SAFE_CHECKS: a safe check available to the attacker (with a queen)
//     triggers the penalty even when too few pieces hit the zone.
//   EV_KS_WIDE_ZONE: the king zone also covers the squares two ranks in front.
// A/B vs. the default rules, 25 ms/move, normal openings (600 games each):
//   one attacker with a queen +2; one attacker if the queen hits the zone -2;
//   safe checks +3; wide zone +18 then -2 in a second run (1200 games +8,
//   LOS 83%; EG suite -5); all three +5; all three with the queen-zone rule -14.
// None proven alone. They only change *when* the SafetyTable penalty
// applies; its size (far smaller than Stockfish 11's king danger in the
// attacking positions that prompted this) is the likelier problem.
// Size tuned next: KingDangerScale plus all weights (Texel, SF labels), then
// A/B vs. the current engine, 600 games from openings:
//   current rules, scale -> 181%: -1.
//   wide zone, scale -> 86%: +6 (LOS 71%).
//   all three rules, scale -> 83%: +17 (LOS 92%), then -6 on new openings
//   (1200 games about +5); EG suite -2.
// Validation loss moved < 0.1% in every case: the training set (2/3
// endgames) had too few king attacks to learn from.
// So 89k king-attack positions were added: tools/datagen attack mode
// (EV_KD_BOOST=400, attacker only), kept when Stockfish 11's phase-weighted
// King safety for a side is <= -0.5 pawn (tools/ks_filter.py), labelled by
// Stockfish 19. Our eval fit them badly (loss 0.057 vs 0.034). Re-tuned on
// the 240k set + these; validation loss now -1.4 to -2%:
//   current rules, scale 131%: +3.   wide zone, scale 71%: -5.
//   all three rules, scale 75%: +12, +13, +22 on three opening sets
//   (1800 games +16, LOS 99%); EG suite -5 (+-14).
// Adopted: all three rules on (queen needed, one attacker enough), with the
// tuned weights.
#ifndef EV_KS_ONE_ATTACKER
#define EV_KS_ONE_ATTACKER 1
#endif
#ifndef EV_KS_SAFE_CHECKS
#define EV_KS_SAFE_CHECKS 1
#endif
#ifndef EV_KS_WIDE_ZONE
#define EV_KS_WIDE_ZONE 1
#endif
// King danger formula and pawn shelter / storm (see danger2 and shelter
// below). EV_KDANGER2 replaces the SafetyTable danger (and the EV_KS_* rules
// except the wide zone, which still sets the squares attackers are counted
// on) with a weighted sum of king-attack signals, squared; EV_SHELTER
// replaces the pawn shield with shelter / storm tables per file.
// Both prompted by a term-by-term comparison with Stockfish 11's eval, where
// king safety was 38% of the disagreement (ideas only, own implementation).
// Tuned (all weights, the formula's own gradient in tools/tuner) on the 329k
// Stockfish-labelled set; validation loss -2.1% (formula), -0.8% (shelter),
// -2.8% (both). A/B vs. the previous engine, 25 ms/move, 600 games each:
//   formula: +37, +48 on new openings (1200 games +42, LOS 100%); EG suite +6.
//   shelter / storm: -32.   both: +8.
// Adopted: the formula. Shelter / storm off.
// Queen-distance king-danger signal (EV_KD_QDIST): our queen far from our
// king while enemy pieces hit the king zone (see KdQueenAway).
#ifndef EV_KD_QDIST
#define EV_KD_QDIST 0
#endif
#ifndef EV_KDANGER2
#define EV_KDANGER2 1
#endif
#ifndef EV_SHELTER
#define EV_SHELTER 0
#endif
// Endgame scale factors (EV_SCALE2; replaces EV_SCALING) and the initiative
// adjustment of the endgame score (EV_INITIATIVE); see scale_info() and
// initiative_signals() below. Ideas from Stockfish 11's eval (own
// implementation; the KPK table is solved here at startup and matched
// Stockfish 11's on 3000 random positions). Tuned with their own gradients in
// tools/tuner on the 329k Stockfish-labelled set; validation loss -12.8%
// (scale), -7.6% (initiative), -14.2% (both). A/B vs. the previous engine,
// 25 ms/move, 600 games each:
//   scale: +22, +2, +26 on three opening sets (1800 games +17, LOS 99%);
//     EG suite +7.
//   initiative: -12.   both: -7 (EG suite +13).
// Adopted: scale factors. Initiative off: the fit gain didn't carry over.
#ifndef EV_SCALE2
#define EV_SCALE2 1
#endif
#ifndef EV_INITIATIVE
#define EV_INITIATIVE 0
#endif
// Stockfish 11-style extras, each its own switch (see threats2(), pieces2(),
// imbalance2(), MobExtra and passed2() below):
//   EV_THREATS2   threats on minors, safe pawn / pawn-push threats, knight or
//                 slider able to hit the queen, restricted squares
//   EV_PIECES2    minor behind a pawn, long-diagonal bishop, minor distance
//                 to own king, trapped rook, rook on a queen's file, weak queen
//   EV_IMBALANCE2 piece-count interaction table (own x own, own x theirs)
//   EV_MOBILITY2  per-count mobility bonus on top of the linear weights
//   EV_PASSED2    passed-pawn path safety and distance from the edge
// Ideas from Stockfish 11's eval (own implementation, own starting values).
// Each tuned alone (all weights, 329k Stockfish-labelled set) and A/B'd vs.
// the engine without them, 600 games from normal openings, 25 ms/move:
//   threats2 +19 (loss -1.9%), pieces2 +13 (-0.8%), mobility2 +11 (-1.3%),
//   passed2 +21 (-0.5%), imbalance2 -16 (-2.3%).
// The four positive ones tuned together (loss -4.3%): +70, +68 on new
// openings (1200 games +69, LOS 100%); EG suite +27. Adopted; imbalance2 off.
#ifndef EV_THREATS2
#define EV_THREATS2 1
#endif
#ifndef EV_PIECES2
#define EV_PIECES2 1
#endif
#ifndef EV_IMBALANCE2
#define EV_IMBALANCE2 0
#endif
#ifndef EV_MOBILITY2
#define EV_MOBILITY2 1
#endif
#ifndef EV_PASSED2
#define EV_PASSED2 1
#endif
// Connected pawns and weak-pawn extras (EV_CONNECTED): see pawn_links() below.
// Idea from Stockfish 11's pawn eval (own implementation). Tuned on the 329k
// Stockfish-labelled set; A/B vs. the engine without it, 600 games from
// normal openings, 25 ms/move:
//   all weights re-tuned with it (validation loss -1.3%): -20.
//   only its own weights tuned (-0.65%; the pawn PSTs already carry most of
//   it, and its values came out small): -9.
// (A control re-tune without it moved the loss by only -0.1%.) Off.
#ifndef EV_CONNECTED
#define EV_CONNECTED 0
#endif
#ifndef EV_SCALING
#define EV_SCALING 0
#endif

inline constexpr bool EvalUsePassed        = EV_PASSED;
inline constexpr bool EvalUseKingSafety    = EV_KING_SAFETY;
inline constexpr bool EvalUsePawnStructure = EV_PAWN_STRUCTURE;
inline constexpr bool EvalUseOutposts      = EV_OUTPOSTS;
inline constexpr bool EvalUseBishops       = EV_BISHOPS;
inline constexpr bool EvalUseRooks         = EV_ROOKS;
inline constexpr bool EvalUseThreats       = EV_THREATS;
inline constexpr bool EvalUseMobilityArea  = EV_MOBILITY_AREA;
inline constexpr bool EvalUseTempo         = EV_TEMPO;
inline constexpr bool EvalUseSpace         = EV_SPACE;
inline constexpr bool EvalUseScaling       = EV_SCALING;
inline constexpr bool EvalUseDanger2       = EV_KDANGER2;
inline constexpr bool EvalUseKdQueenDist   = EV_KD_QDIST;
inline constexpr bool EvalUseScale2        = EV_SCALE2;
inline constexpr bool EvalUseConnected     = EV_CONNECTED;
inline constexpr bool EvalUseThreats2      = EV_THREATS2;
inline constexpr bool EvalUsePieces2       = EV_PIECES2;
inline constexpr bool EvalUseImbalance2    = EV_IMBALANCE2;
inline constexpr bool EvalUseMobility2     = EV_MOBILITY2;
inline constexpr bool EvalUsePassed2       = EV_PASSED2;
inline constexpr bool EvalUseInitiative    = EV_INITIATIVE;
inline constexpr bool EvalUseShelter       = EV_SHELTER;
inline constexpr int  KsOneAttacker        = EV_KS_ONE_ATTACKER;
inline constexpr bool KsSafeChecks         = EV_KS_SAFE_CHECKS;
inline constexpr bool KsWideZone           = EV_KS_WIDE_ZONE;
inline constexpr bool EvalUsePassedExtra   = EV_PASSED_EXTRA;   // outside + unstoppable passed pawns
inline constexpr bool EvalUseCandidate     = EV_CANDIDATE;      // candidate passed pawns
inline constexpr bool EvalUsePawnThreats   = EV_PAWN_THREATS;   // pieces / king attacking undefended pawns
inline constexpr bool EvalUseRookPassers   = EV_ROOK_PASSERS;   // rooks behind / in front of enemy passers
inline constexpr bool EvalUseWeakPawnThreats = EV_WEAK_PAWN_THREATS;  // minors / rooks hitting pawns no pawn defends
inline constexpr bool EvalUseBadBishop3    = EV_BAD_BISHOP3;    // extra penalty per fixed own pawn on the bishop's colour
inline constexpr bool EvalUseImbalance     = EV_IMBALANCE;      // knight / rook value by own pawn count

// ---- Weights ---------------------------------------------------------------
// Passed pawns, by relative rank (rank 2 = index 1 ... rank 7 = index 6).
// Started from the author's earlier engine, then tuned. They overlap with the
// pawn PSTs (both reward advanced pawns), so only their sum is meaningful.
inline constexpr Score PassedMG[8] = {0, 1, -16, -1, 14, 32, 55, 0};
inline constexpr Score PassedEG[8] = {0, 21, 16, 25, 37, 58, 110, 0};
inline constexpr MgEg PassedProtected = {22, 10};   // defended by own pawn
inline constexpr MgEg PassedBlocked = {-6, -3}; // stop square occupied
inline constexpr Score PassedFreePathEG[8] = {0, -2, 0, 1, 12, 25, 77, 0};
inline constexpr Score PassedKingDistEG = 4;           // x (rank-2) x (2*their king dist - our king dist)
inline constexpr MgEg RookBehindPasser = {4, 21};

// King safety (EV_KDANGER2 off): attack units -> penalty (mg), when the attacker has a queen (rules: EV_KS_*).
inline constexpr int KingAttackWeight[6] = {0, 2, 2, 3, 5, 0};  // per attacked zone square
inline constexpr Score KingDangerScale = 75;  // % applied to SafetyTable (tuner slot)
inline constexpr int SafeCheckUnits[6]   = {0, 3, 2, 4, 6, 0};  // N, B, R, Q safe check available
inline constexpr Score ShieldRank3 = -10, ShieldMissing = -24, ShieldOpenFile = -15;  // mg, per file
// Classic attack-unit table (chessprogramming.org "King Safety").
inline constexpr Score SafetyTable[64] = {
      0,   0,   1,   2,   3,   5,   7,   9,  12,  15,  18,  22,  26,  30,  35,  39,
     44,  50,  56,  62,  68,  75,  82,  85,  89,  97, 105, 113, 122, 131, 140, 150,
    169, 180, 191, 202, 213, 225, 237, 248, 260, 272, 283, 295, 307, 319, 330, 342,
    354, 366, 377, 389, 401, 412, 424, 436, 448, 459, 471, 483, 494, 500, 500, 500,
};

// Pawn structure (per pawn).
inline constexpr MgEg IsolatedPawn = {-11, -15};
inline constexpr MgEg DoubledPawn = {-2, -10};
inline constexpr MgEg BackwardPawn = {-8, -10};

// Outposts (protected by own pawn, can never be attacked by an enemy pawn).
inline constexpr MgEg KnightOutpost = {25, 15};
inline constexpr MgEg BishopOutpost = {12, 6};

// Bishops.
inline constexpr MgEg BishopPair = {33, 44};
inline constexpr MgEg BadBishopPerPawn = {-1, -3};  // own pawns on the bishop's colour

// Rooks.
inline constexpr MgEg RookOpenFile = {36, 4};
inline constexpr MgEg RookSemiOpenFile = {17, 8};
inline constexpr MgEg RookOnSeventh = {-16, 7};

// Threats (bonus for the attacking side).
inline constexpr MgEg ThreatByPawn = {29, 77};  // pawn attacks a piece
inline constexpr MgEg ThreatByMinor = {45, 26};  // knight/bishop attacks rook/queen
inline constexpr MgEg ThreatByRook = {46, 5};  // rook attacks queen
inline constexpr MgEg HangingPiece = {13, 14};  // attacked and undefended

// Endgame terms. Hand-set, then Texel-tuned with every older weight frozen
// (tools/tuner ... only_new=1).
inline constexpr Score PassedOutsideEG = 18;        // passer on a/b/g/h, 2+ files from every enemy pawn
inline constexpr Score UnstoppableEG = 400;         // opponent has only king + pawns and can't catch it
inline constexpr Score CandidateEG[8] = {0, 5, 6, 11, 16, 26, 0, 0};  // by relative rank
#ifndef EV_PAWN_THREATS_MG
#define EV_PAWN_THREATS_MG 8
#endif
inline constexpr MgEg ThreatOnPawn = {17, 27};        // minor / rook attacks an undefended pawn
inline constexpr Score KingThreatOnPawnEG = 53;     // king attacks an undefended pawn
inline constexpr MgEg RookBehindEnemyPasser = {0, 18};
inline constexpr MgEg RookInFrontOfEnemyPasser = {0, -8};

// Second-round terms (starting values; tuned on Stockfish labels).
inline constexpr MgEg ThreatOnWeakPawn = {0, 9};     // minor / rook attacks a pawn not defended by a pawn
inline constexpr MgEg BadBishopFixedPawn = {-4, -7}; // per own pawn on the bishop's colour that can't advance
inline constexpr MgEg KnightPerPawn = {3, 3};          // per knight, per own pawn above 5 (knights like closed positions)
inline constexpr MgEg RookPerPawn = {11, -10};          // per rook, per own pawn above 5 (rooks like open ones)

// King danger formula (EV_KDANGER2), for one king. danger = sum of
// KdWeight[i] * signal i (signals: KdFeature below); penalty = danger^2 / 1024
// in the middlegame and danger * KdEgSlope / 64 in the endgame, when danger > 0.
inline constexpr int KdWeight[18] = {27, 3, 7, -12, 16, 14, 142, 96, 183, 147, 27, 9, -17, 64, -37, -159, 194, 0};
inline constexpr int KdEgSlope = 8;

// Pawn shelter / storm (EV_SHELTER), mg, per file of the three in front of
// the king. Index [d][r]: d = file distance from the board edge (0-3),
// r = relative rank of our rearmost pawn there (0 = none) / the enemy's
// frontmost pawn there (0 = none). A storm pawn directly blocked by our
// shelter pawn uses BlockedStorm[r].
inline constexpr Score ShelterMG[4][7] = {{-10, 35, 25, 5, 0, 0, 0}, {-10, 35, 25, 5, 0, 0, 0}, {-10, 35, 25, 5, 0, 0, 0}, {-10, 35, 25, 5, 0, 0, 0}};
inline constexpr Score StormMG[4][8] = {{0, 0, -45, -25, -10, -4, 0, 0}, {0, 0, -45, -25, -10, -4, 0, 0}, {0, 0, -45, -25, -10, -4, 0, 0}, {0, 0, -45, -25, -10, -4, 0, 0}};
inline constexpr Score BlockedStormMG[8] = {0, 0, -10, -5, 0, 0, 0, 0};
inline constexpr Score KingPawnDistEG = -8;  // per square to the nearest own pawn

// Initiative (EV_INITIATIVE): c = sum InitWeight[i] * signal i (InitFeature
// below); the endgame score moves by c towards the side that is ahead, but a
// negative c can at most bring it to 0.
inline constexpr int InitWeight[8] = {10, 10, 20, 8, 10, 40, -40, -100};

// Endgame scale factors (EV_SCALE2), out of 64, for the side that is ahead
// (ScaleCat below says which applies):
//   [0] + [1] * its pawns      [0] + [2] * its pawns (opposite bishops, other pieces too)
//   [3] + [4] * passed pawns (bishops of opposite colour only)
//   [5] + [6] * minors facing a lone queen      [7] no pawns, <= a minor ahead
//   [8] KRKP drawish      [9] KQKP drawish (7th-rank a/c/f/h pawn)
inline constexpr int ScaleParam[10] = {24, 12, 8, 6, 8, 57, 20, 5, 7, 5};

// Connected pawns (EV_CONNECTED), by the pawn's relative rank.
inline constexpr Score PhalanxMG[8] = {0, 2, 5, 8, 15, 30, 50, 0};    // own pawn beside it
inline constexpr Score PhalanxEG[8] = {0, 0, 2, 5, 12, 25, 40, 0};
inline constexpr Score SupportedMG[8] = {0, 0, 6, 5, 10, 20, 35, 0};  // defended by an own pawn
inline constexpr Score SupportedEG[8] = {0, 0, 3, 4, 10, 20, 35, 0};
inline constexpr MgEg SupportCount = {5, 3};        // per defending own pawn
inline constexpr MgEg ConnectedOpposed = {-5, -5};  // connected, but an enemy pawn is ahead on its file
inline constexpr MgEg WeakUnopposed = {-10, -15};   // isolated or backward, no enemy pawn ahead on its file
inline constexpr MgEg WeakLever = {-5, -40};        // attacked by two enemy pawns, no own pawn defends it

// EV_THREATS2, by threats2() signal: minor hits an enemy minor not defended by
// a pawn; rook hits one; our safe pawns hit a piece; a safe pawn push would
// hit a piece; our knight can reach a safe square hitting the queen; a
// bishop / rook likewise (square attacked twice by us); restricted squares.
inline constexpr MgEg Threats2W[7] = {{19, 26}, {19, 28}, {34, -37}, {19, 16}, {7, -3}, {14, 1}, {5, 2}};
// EV_PIECES2, by pieces2() signal: minor with a pawn right in front; bishop
// seeing two centre squares through pawns; knight / bishop distance to own
// king (per square); trapped rook; trapped rook without castling rights (extra);
// rook on a file with a queen; our queen pinned / exposed to a discovered attack.
inline constexpr MgEg Pieces2W[8] = {{4, 10}, {20, 17}, {-4, -3}, {-4, -1}, {11, 2}, {-43, -16}, {7, 10}, {-18, 28}};
// EV_IMBALANCE2: counts (0 bishop pair, 1 P, 2 N, 3 B, 4 R, 5 Q); weights for
// own_i * own_j (j <= i, 21) then own_i * theirs_j (j < i, 15).
inline constexpr MgEg Imbalance2W[36] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
// EV_MOBILITY2: extra bonus by number of mobility-area squares: N 0-8, B 0-13,
// R 0-14, Q 0-27 (index offsets 0, 9, 23, 38).
inline constexpr MgEg MobExtra[66] = {{-18, 0}, {0, 77}, {7, 105}, {9, 109}, {10, 103}, {9, 96}, {4, 88}, {-1, 79}, {-9, 67}, {-5, 13}, {6, 43}, {18, 64}, {17, 77}, {18, 88}, {16, 92}, {14, 95}, {10, 98}, {5, 101}, {-1, 101}, {0, 98}, {-10, 100}, {0, 100}, {0, 102}, {0, 0}, {-58, 82}, {-45, 114}, {-42, 114}, {-38, 107}, {-37, 103}, {-37, 101}, {-36, 92}, {-35, 85}, {-37, 82}, {-37, 74}, {-44, 69}, {-43, 61}, {-48, 56}, {-48, 48}, {0, 0}, {0, 0}, {17, 0}, {16, 0}, {23, 0}, {26, 0}, {20, 98}, {16, 111}, {14, 127}, {12, 139}, {8, 150}, {3, 161}, {-2, 171}, {-6, 177}, {-17, 182}, {-14, 182}, {-22, 192}, {-29, 195}, {-33, 198}, {-36, 199}, {-27, 199}, {0, 171}, {0, 177}, {0, 164}, {0, 181}, {0, 145}, {0, 0}, {0, 137}};
// EV_PASSED2 (per passed pawn, relative rank index): path to promotion not
// attacked or occupied by the enemy [0-7]; else stop square safe [8-15]; every
// path square defended by us [16-23]; per file of distance from the edge [24].
inline constexpr MgEg Passed2W[25] = {{0, 0}, {0, 0}, {0, 0}, {0, 15}, {0, 28}, {0, 53}, {0, 70}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 1}, {5, 5}, {24, 16}, {0, 20}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 10}, {0, 50}, {0, 64}, {0, 89}, {0, 0}, {-5, -5}};

inline constexpr Score Tempo = 18;       // side to move
inline constexpr Score SpacePerSquare = 4;  // mg

// ---- Bitboard helpers (a1 = bit 0, h8 = bit 63) -----------------------------
namespace bb {
using U64 = std::uint64_t;
inline constexpr U64 FileA = 0x0101010101010101ULL;
inline constexpr U64 FileH = FileA << 7;
inline constexpr U64 Rank1 = 0xFFULL;
inline constexpr U64 DarkSquares = 0xAA55AA55AA55AA55ULL;  // a1 is dark

constexpr U64 file_mask(int f) { return FileA << f; }
constexpr U64 rank_mask(int r) { return Rank1 << (8 * r); }
constexpr U64 adjacent_files(int f) {
    return (f > 0 ? file_mask(f - 1) : 0) | (f < 7 ? file_mask(f + 1) : 0);
}
// Ranks strictly in front of rank r, from `white`'s point of view.
constexpr U64 ranks_ahead(int r, bool white) {
    U64 m = 0;
    for (int i = 0; i < 8; ++i)
        if (white ? i > r : i < r) m |= rank_mask(i);
    return m;
}
constexpr int rel_rank(int sq, bool white) { return white ? sq / 8 : 7 - sq / 8; }

struct Masks {
    U64 forward_file[2][64]{};  // same file, ahead
    U64 passed[2][64]{};        // same + adjacent files, ahead
    U64 attack_span[2][64]{};   // adjacent files, ahead (where enemy pawns could attack from)
};
constexpr Masks build_masks() {
    Masks m{};
    for (int c = 0; c < 2; ++c)
        for (int sq = 0; sq < 64; ++sq) {
            const U64 ahead = ranks_ahead(sq / 8, c == 0);
            m.forward_file[c][sq] = file_mask(sq % 8) & ahead;
            m.attack_span[c][sq] = adjacent_files(sq % 8) & ahead;
            m.passed[c][sq] = m.forward_file[c][sq] | m.attack_span[c][sq];
        }
    return m;
}
inline constexpr Masks M = build_masks();

inline U64 pawn_attacks(U64 pawns, bool white) {
    return white ? (((pawns << 7) & ~FileH) | ((pawns << 9) & ~FileA))
                 : (((pawns >> 9) & ~FileH) | ((pawns >> 7) & ~FileA));
}
inline int popcount(U64 x) { return std::popcount(x); }
inline int lsb(U64 x) { return std::countr_zero(x); }
inline int distance(int a, int b) {
    return std::max(std::abs(a % 8 - b % 8), std::abs(a / 8 - b / 8));
}
}  // namespace bb

// ---- Attack information, both sides ---------------------------------------
struct AttackInfo {
    bb::U64 by[2][6]{};  // [color][piece type] squares attacked
    bb::U64 all[2]{};
    bb::U64 pieces[2][6]{};
    bb::U64 occ[2]{};
    // King attack bookkeeping: attacker side's pieces hitting the enemy king zone.
    int king_attackers[2]{};  // indexed by attacking side
    bb::U64 zone[2]{};        // king zone of each side (the squares attackers are counted on)
    int king_units[2]{};
    int zone_attackers[2][6]{};  // [attacking side][piece type]: pieces hitting the enemy king zone
    int zone_hits[2]{};          // attacked king-zone squares, summed over the attacking pieces
    bb::U64 twice[2]{};          // squares attacked at least twice
    bool castle[2][2]{};         // [side][0 = king side, 1 = queen side] castling rights
    // Slider mobility (weight * attacked squares) and mobility-area correction.
    MgEg slider[2]{};
    MgEg mobility_adjust[2]{};
    int stm = 0;  // side to move (0 = white)
};

// MobExtra index for piece type t (1 N .. 4 Q) with n mobility-area squares.
inline int mob2_index(int t, int n) {
    constexpr int base[5] = {0, 0, 9, 23, 38}, cap[5] = {0, 8, 13, 14, 27};
    return base[t] + std::min(n, cap[t]);
}

inline AttackInfo gather_attacks(const chess::Board& b) {
    using namespace bb;
    AttackInfo ai;
    ai.stm = b.sideToMove() == chess::Color::WHITE ? 0 : 1;
    {
        using CS = chess::Board::CastlingRights::Side;
        const auto cr = b.castlingRights();
        for (int c = 0; c < 2; ++c) {
            const chess::Color col = c == 0 ? chess::Color::WHITE : chess::Color::BLACK;
            ai.castle[c][0] = cr.has(col, CS::KING_SIDE);
            ai.castle[c][1] = cr.has(col, CS::QUEEN_SIDE);
        }
    }
    const U64 occ = b.occ().getBits();
    for (int c = 0; c < 2; ++c) {
        const chess::Color col = c == 0 ? chess::Color::WHITE : chess::Color::BLACK;
        for (int t = 0; t < 6; ++t)
            ai.pieces[c][t] = b.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(t)), col).getBits();
        ai.occ[c] = b.us(col).getBits();
    }
    U64 zone[2];
    for (int c = 0; c < 2; ++c) {
        const int ksq = lsb(ai.pieces[c][5]);
        zone[c] = chess::attacks::king(chess::Square(ksq)).getBits() | (1ULL << ksq);
        if (KsWideZone) zone[c] |= c == 0 ? (zone[c] << 8) : (zone[c] >> 8);  // + two ranks in front
        ai.zone[c] = zone[c];
    }
    for (int c = 0; c < 2; ++c) {
        const bool white = (c == 0);
        const U64 excluded = ai.pieces[c][0] | pawn_attacks(ai.pieces[c ^ 1][0], !white);  // mobility area
        ai.by[c][0] = pawn_attacks(ai.pieces[c][0], white);
        ai.by[c][5] = chess::attacks::king(chess::Square(lsb(ai.pieces[c][5]))).getBits();
        {
            const U64 p = ai.pieces[c][0];
            const U64 left = white ? ((p << 7) & ~FileH) : ((p >> 9) & ~FileH);
            const U64 right = white ? ((p << 9) & ~FileA) : ((p >> 7) & ~FileA);
            ai.twice[c] = (left & right) | (ai.by[c][0] & ai.by[c][5]);
        }
        U64 seen = ai.by[c][0] | ai.by[c][5];
        const MobilityWeight* mw[5] = {nullptr, &KnightMobility, &BishopMobility, &RookMobility, &QueenMobility};
        for (int t = 1; t <= 4; ++t) {
            U64 pcs = ai.pieces[c][t];
            while (pcs) {
                const int sq = lsb(pcs);
                pcs &= pcs - 1;
                const chess::Square s(sq);
                U64 a = 0;
                switch (t) {
                    case 1: a = chess::attacks::knight(s).getBits(); break;
                    case 2: a = chess::attacks::bishop(s, chess::Bitboard(occ)).getBits(); break;
                    case 3: a = chess::attacks::rook(s, chess::Bitboard(occ)).getBits(); break;
                    case 4: a = chess::attacks::queen(s, chess::Bitboard(occ)).getBits(); break;
                }
                ai.by[c][t] |= a;
                ai.twice[c] |= seen & a;
                seen |= a;
                if (t >= 2) {
                    ai.slider[c].mg += mw[t]->mg * popcount(a);
                    ai.slider[c].eg += mw[t]->eg * popcount(a);
                }
                if (const U64 hit = a & zone[c ^ 1]) {
                    ++ai.king_attackers[c];
                    ++ai.zone_attackers[c][t];
                    ai.zone_hits[c] += popcount(hit);
                    ai.king_units[c] += KingAttackWeight[t] * popcount(hit);
                }
                if (EvalUseMobilityArea) {
                    const int n = popcount(a & excluded);
                    ai.mobility_adjust[c].mg -= mw[t]->mg * n;
                    ai.mobility_adjust[c].eg -= mw[t]->eg * n;
                }
                if (EvalUseMobility2) {
                    const MgEg& e = MobExtra[mob2_index(t, popcount(a & ~excluded))];
                    ai.mobility_adjust[c].mg += e.mg;
                    ai.mobility_adjust[c].eg += e.eg;
                }
            }
        }
        for (int t = 0; t < 6; ++t) ai.all[c] |= ai.by[c][t];
    }
    return ai;
}

// ---- Per-side positional score (that side's point of view) ----------------
enum Term {
    TermPassed, TermKingSafety, TermStructure, TermOutposts, TermBishops, TermRooks,
    TermThreats, TermMobilityArea, TermSpace, TermCount
};
inline constexpr const char* TermNames[TermCount] = {
    "passed", "king safety", "pawn struct", "outposts", "bishops", "rooks",
    "threats", "mob. area", "space",
};
struct TermScores {
    MgEg t[TermCount]{};
    MgEg total() const {
        MgEg s;
        for (const auto& x : t) { s.mg += x.mg; s.eg += x.eg; }
        return s;
    }
};

// Side c's passed pawns (no enemy pawn ahead on its own or adjacent files,
// frontmost of its file).
inline bb::U64 passed_pawns(const AttackInfo& ai, int c) {
    using namespace bb;
    const U64 ours = ai.pieces[c][0], theirs = ai.pieces[c ^ 1][0];
    U64 res = 0, p = ours;
    while (p) {
        const int sq = lsb(p);
        p &= p - 1;
        if ((M.passed[c][sq] & theirs) == 0 && (M.forward_file[c][sq] & ours) == 0) res |= 1ULL << sq;
    }
    return res;
}

// Connected-pawn signals for side c's pawn on sq (EV_CONNECTED): calls
// f(kind, index, count): kind 0 = phalanx at rank index, 1 = supported at
// rank index, 2 = number of own defenders, 3 = connected but opposed,
// 4 = weak (isolated / backward) and unopposed, 5 = lever-attacked twice and
// undefended.
template <typename F>
inline void pawn_links(const AttackInfo& ai, int c, int sq, F&& f) {
    using namespace bb;
    const bool white = (c == 0);
    const int them = c ^ 1;
    const U64 ours = ai.pieces[c][0], theirs = ai.pieces[them][0];
    const int file = sq % 8, r = rel_rank(sq, white);
    const U64 bit = 1ULL << sq;
    const U64 rank = rank_mask(sq / 8);
    const bool phalanx = (adjacent_files(file) & rank & ours) != 0;
    const U64 defenders = pawn_attacks(bit, !white) & ours;  // own pawns that defend sq
    const bool opposed = (M.forward_file[c][sq] & theirs) != 0;
    if (phalanx) f(0, r, 1);
    if (defenders) {
        f(1, r, 1);
        f(2, 0, popcount(defenders));
    }
    if ((phalanx || defenders) && opposed) f(3, 0, 1);
    const bool isolated = (adjacent_files(file) & ours) == 0;
    bool backward = false;
    if (!isolated) {
        const int stop = white ? sq + 8 : sq - 8;
        backward = (adjacent_files(file) & ~ranks_ahead(sq / 8, white) & ours) == 0 && stop >= 0 && stop < 64 &&
                   (ai.by[them][0] & (1ULL << stop));
    }
    if ((isolated || backward) && !opposed) f(4, 0, 1);
    const U64 attackers = pawn_attacks(bit, white) & theirs;  // enemy pawns attacking sq
    if (popcount(attackers) == 2 && !defenders) f(5, 0, 1);
}

// King danger for side c's king, unscaled (SafetyTable value, >= 0): attack
// units from enemy pieces hitting the king zone plus safe checks. Only when
// the attacker has a queen, and (by default) 2+ attacking pieces; EV_KS_*
// relax that (see the switches). The tuner uses it as the feature whose
// weight is KingDangerScale.
inline Score danger_unscaled(const AttackInfo& ai, int c) {
    using namespace bb;
    const int them = c ^ 1;
    if (!ai.pieces[them][4]) return 0;
    const int our_k = lsb(ai.pieces[c][5]);
    const U64 occ = ai.occ[0] | ai.occ[1];
    int units = ai.king_units[them];
    // Safe checks the opponent could give.
    const chess::Square k(our_k);
    const U64 safe = ~ai.all[c] & ~ai.occ[them];
    const U64 nchk = chess::attacks::knight(k).getBits();
    const U64 bchk = chess::attacks::bishop(k, chess::Bitboard(occ)).getBits();
    const U64 rchk = chess::attacks::rook(k, chess::Bitboard(occ)).getBits();
    bool check = false;
    if (nchk & ai.by[them][1] & safe) { units += SafeCheckUnits[1]; check = true; }
    if (bchk & ai.by[them][2] & safe) { units += SafeCheckUnits[2]; check = true; }
    if (rchk & ai.by[them][3] & safe) { units += SafeCheckUnits[3]; check = true; }
    if ((bchk | rchk) & ai.by[them][4] & safe) { units += SafeCheckUnits[4]; check = true; }

    int needed = 2;
    if (KsOneAttacker == 1) needed = 1;
    if (KsOneAttacker == 2 && (ai.by[them][4] & ai.zone[c])) needed = 1;
    const bool applies = ai.king_attackers[them] >= needed || (KsSafeChecks && check);
    return applies ? SafetyTable[std::min(units, 63)] : 0;
}

// The middlegame king-danger penalty actually applied.
// Data generation "attack mode" (tools/datagen, EV_KD_BOOST = percent,
// default 100 = off and compiled out): the danger to king kd_boost_victim is
// multiplied, so a search using it overrates attacks on that king. datagen
// gives it only to the attacking side's searches, steering games into
// king-attack positions, which the normal training data lacks.
#ifndef EV_KD_BOOST
#define EV_KD_BOOST 100
#endif
#if EV_KD_BOOST != 100
inline int kd_boost_victim = -1;  // colour whose king danger is boosted, -1 = none
#endif
inline Score danger_penalty(const AttackInfo& ai, int c) {
    const Score d = danger_unscaled(ai, c) * KingDangerScale / 100;
#if EV_KD_BOOST != 100
    if (c == kd_boost_victim) return d * EV_KD_BOOST / 100;
#endif
    return d;
}

// ---- King danger formula (EV_KDANGER2) --------------------------------------
// Signals for the danger to side c's king, weighted by KdWeight.
enum KdFeature {
    KdKnights, KdBishops, KdRooks, KdQueens,  // enemy pieces of that type hitting the king zone
    KdZoneHits,        // attacked king-zone squares, summed over those pieces
    KdWeakRing,        // squares next to the king the enemy attacks that we defend at most once, by K or Q only
    KdSafeKnightCheck, KdSafeBishopCheck, KdSafeRookCheck, KdSafeQueenCheck,  // 0/1 each
    KdUnsafeChecks,    // piece types with a check available, but only on unsafe squares
    KdFlankAttacks,    // enemy attacks on our camp (own 5 ranks) on the king's flank, double attacks twice
    KdFlankDefense,    // our attacks on the same squares
    KdPinned,          // our pieces pinned to the king
    KdKnightDefender,  // one of our knights guards a square next to the king (0/1)
    KdNoQueen,         // the enemy has no queen (0/1)
    KdBias,            // always 1
    KdQueenAway,       // (EV_KD_QDIST) our queen's distance from our king x min(enemy pieces hitting the zone, 2)
    KdCount
};
static_assert(KdCount == 18);

struct KdSignals {
    int x[KdCount]{};
};

// Squares strictly between two aligned squares (empty if not aligned).
struct BetweenTable {
    bb::U64 t[64][64]{};
    BetweenTable() {
        for (int a = 0; a < 64; ++a)
            for (int b = 0; b < 64; ++b) {
                if (a == b) continue;
                const bb::U64 ab = 1ULL << a, bbit = 1ULL << b;
                const chess::Square sa(a), sb(b);
                const int df = b % 8 - a % 8, dr = b / 8 - a / 8;
                if (df == 0 || dr == 0)
                    t[a][b] = chess::attacks::rook(sa, chess::Bitboard(bbit)).getBits() &
                              chess::attacks::rook(sb, chess::Bitboard(ab)).getBits();
                else if (df == dr || df == -dr)
                    t[a][b] = chess::attacks::bishop(sa, chess::Bitboard(bbit)).getBits() &
                              chess::attacks::bishop(sb, chess::Bitboard(ab)).getBits();
            }
    }
};
inline const BetweenTable& between_table() {
    static const BetweenTable table;
    return table;
}

inline KdSignals danger2_signals(const AttackInfo& ai, int c) {
    using namespace bb;
    KdSignals k;
    const int them = c ^ 1;
    const bool white = (c == 0);
    const int ksq = lsb(ai.pieces[c][5]);
    const chess::Square ks(ksq);
    const U64 occ = ai.occ[0] | ai.occ[1];
    const U64 ring = ai.by[c][5];

    for (int t = 1; t <= 4; ++t) k.x[KdKnights + t - 1] = ai.zone_attackers[them][t];
    k.x[KdZoneHits] = ai.zone_hits[them];

    // Weak: attacked by them, not defended twice, and undefended or defended only by our king / queen.
    const U64 weak = ai.all[them] & ~ai.twice[c] & (~ai.all[c] | ai.by[c][4] | ai.by[c][5]);
    k.x[KdWeakRing] = popcount(weak & ring);

    // Checks: safe = not occupied by them, and not defended by us (or weak and attacked twice by them).
    const U64 safe = ~ai.occ[them] & (~ai.all[c] | (weak & ai.twice[them]));
    const U64 nchk = chess::attacks::knight(ks).getBits() & ai.by[them][1];
    const U64 bline = chess::attacks::bishop(ks, chess::Bitboard(occ)).getBits();
    const U64 rline = chess::attacks::rook(ks, chess::Bitboard(occ)).getBits();
    const U64 bchk = bline & ai.by[them][2], rchk = rline & ai.by[them][3];
    const U64 qchk = (bline | rline) & ai.by[them][4] & ~ai.by[c][4];
    const U64 chk[4] = {nchk, bchk, rchk, qchk};
    for (int i = 0; i < 4; ++i) {
        const U64 cands = chk[i] & ~ai.occ[them];
        if (cands & safe) k.x[KdSafeKnightCheck + i] = 1;
        else if (cands) ++k.x[KdUnsafeChecks];
    }

    // Flank: files a-d, c-f or e-h by the king's file; camp = our first five ranks.
    const int kf = ksq % 8;
    const U64 flank = kf <= 2 ? (file_mask(0) | file_mask(1) | file_mask(2) | file_mask(3))
                    : kf <= 4 ? (file_mask(2) | file_mask(3) | file_mask(4) | file_mask(5))
                              : (file_mask(4) | file_mask(5) | file_mask(6) | file_mask(7));
    const U64 camp = white ? (rank_mask(0) | rank_mask(1) | rank_mask(2) | rank_mask(3) | rank_mask(4))
                           : (rank_mask(7) | rank_mask(6) | rank_mask(5) | rank_mask(4) | rank_mask(3));
    const U64 zone = flank & camp;
    k.x[KdFlankAttacks] = popcount(ai.all[them] & zone) + popcount(ai.twice[them] & zone);
    k.x[KdFlankDefense] = popcount(ai.all[c] & zone);

    // Pins: enemy sliders lined up with our king with exactly one piece, ours, in between.
    U64 snipers = (chess::attacks::bishop(ks, chess::Bitboard(0)).getBits() & (ai.pieces[them][2] | ai.pieces[them][4])) |
                  (chess::attacks::rook(ks, chess::Bitboard(0)).getBits() & (ai.pieces[them][3] | ai.pieces[them][4]));
    while (snipers) {
        const int s = lsb(snipers);
        snipers &= snipers - 1;
        const U64 between = between_table().t[ksq][s] & occ;
        if (between && (between & (between - 1)) == 0 && (between & ai.occ[c])) ++k.x[KdPinned];
    }

    k.x[KdKnightDefender] = (ai.by[c][1] & ring) ? 1 : 0;
    k.x[KdNoQueen] = ai.pieces[them][4] ? 0 : 1;
    k.x[KdBias] = 1;
    if (EvalUseKdQueenDist && ai.pieces[c][4])
        k.x[KdQueenAway] = distance(lsb(ai.pieces[c][4]), ksq) * std::min(ai.king_attackers[them], 2);
    return k;
}

inline int danger2_value(const KdSignals& k) {
    int d = 0;
    for (int i = 0; i < KdCount; ++i) d += KdWeight[i] * k.x[i];
    return d;
}

// Penalty for side c's king (positive = bad for c).
inline MgEg danger2_penalty(const AttackInfo& ai, int c) {
    const int d = danger2_value(danger2_signals(ai, c));
    if (d <= 0) return {0, 0};
    return {d * d / 1024, d * KdEgSlope / 64};
}

// ---- Pawn shelter / storm (EV_SHELTER) --------------------------------------
// Table entries used for a king of side c standing on ksq: calls
// f(kind, index) for each of the three files (kind 0 = ShelterMG[d][r] with
// index d * 7 + r, 1 = StormMG[d][r] with index d * 8 + r, 2 = BlockedStormMG[r]).
template <typename F>
inline void shelter_entries(const AttackInfo& ai, int c, int ksq, F&& f) {
    using namespace bb;
    const bool white = (c == 0);
    const int them = c ^ 1;
    const int kr = rel_rank(ksq, white);
    const U64 notbehind = white ? ~0ULL << (8 * kr) : ~0ULL >> (8 * kr);  // level with or in front of the king
    const int center = std::clamp(ksq % 8, 1, 6);
    for (int file = center - 1; file <= center + 1; ++file) {
        const U64 m = file_mask(file) & notbehind;
        const U64 ours = ai.pieces[c][0] & m, theirs = ai.pieces[them][0] & m;
        // Our rearmost pawn / their frontmost one (the one nearest our back rank).
        const int our_r = ours ? rel_rank(white ? lsb(ours) : 63 - std::countl_zero(ours), white) : 0;
        const int their_r = theirs ? rel_rank(white ? lsb(theirs) : 63 - std::countl_zero(theirs), white) : 0;
        const int d = std::min(file, 7 - file);
        f(0, d * 7 + std::min(our_r, 6));
        if (our_r && our_r == their_r - 1) f(2, their_r);
        else f(1, d * 8 + their_r);
    }
}

inline Score shelter_at(const AttackInfo& ai, int c, int ksq) {
    Score s = 0;
    shelter_entries(ai, c, ksq, [&](int kind, int i) {
        s += kind == 0 ? ShelterMG[i / 7][i % 7] : kind == 1 ? StormMG[i / 8][i % 8] : BlockedStormMG[i];
    });
    return s;
}

// The square the shelter is scored from: the king's square, or the square it
// can still castle to, whichever gives the better shelter.
inline int shelter_square(const AttackInfo& ai, int c) {
    const int ksq = bb::lsb(ai.pieces[c][5]);
    int best = ksq;
    Score best_s = shelter_at(ai, c, ksq);
    const int back = c == 0 ? 0 : 56;
    for (int side = 0; side < 2; ++side)
        if (ai.castle[c][side]) {
            const int sq = back + (side == 0 ? 6 : 2);
            const Score v = shelter_at(ai, c, sq);
            if (v > best_s) { best_s = v; best = sq; }
        }
    return best;
}

// Distance from side c's king to its nearest pawn (0 without pawns).
struct DistRings {
    bb::U64 ring[64][8]{};  // squares at exactly distance d
    constexpr DistRings() {
        for (int a = 0; a < 64; ++a)
            for (int b = 0; b < 64; ++b) {
                const int df = a % 8 - b % 8, dr = a / 8 - b / 8;
                const int d = std::max(df < 0 ? -df : df, dr < 0 ? -dr : dr);
                ring[a][d] |= 1ULL << b;
            }
    }
};
inline constexpr DistRings Rings{};

inline int king_pawn_distance(const AttackInfo& ai, int c) {
    const bb::U64 pawns = ai.pieces[c][0];
    if (!pawns) return 0;
    const int ksq = bb::lsb(ai.pieces[c][5]);
    for (int d = 1; d < 8; ++d)
        if (pawns & Rings.ring[ksq][d]) return d;
    return 0;
}

// ---- EV_THREATS2 / EV_PIECES2 / EV_IMBALANCE2 / EV_PASSED2 signals ----------
// Each calls f(index, count) for side c (its own point of view).
template <typename F>
inline void threats2(const AttackInfo& ai, int c, F&& f) {
    using namespace bb;
    const bool white = (c == 0);
    const int them = c ^ 1;
    const U64 occ = ai.occ[0] | ai.occ[1];
    const U64 minors_them = ai.pieces[them][1] | ai.pieces[them][2];
    const U64 pieces_them = minors_them | ai.pieces[them][3] | ai.pieces[them][4];
    const U64 no_pawn_guard = ~ai.by[them][0];
    f(0, popcount((ai.by[c][1] | ai.by[c][2]) & minors_them & no_pawn_guard));
    f(1, popcount(ai.by[c][3] & minors_them & no_pawn_guard));
    const U64 safe = ~ai.all[them] | ai.all[c];
    f(2, popcount(pawn_attacks(ai.pieces[c][0] & safe, white) & pieces_them));
    // Pawn pushes (single, and double from the start rank) to safe empty squares.
    const U64 rank3 = white ? rank_mask(2) : rank_mask(5);
    U64 push = (white ? ai.pieces[c][0] << 8 : ai.pieces[c][0] >> 8) & ~occ;
    push |= (white ? (push & rank3) << 8 : (push & rank3) >> 8) & ~occ;
    push &= ~ai.by[them][0] & safe;
    f(3, popcount(pawn_attacks(push, white) & pieces_them));
    // Hitting the queen next move (one enemy queen).
    const U64 strongly = ai.by[them][0] | (ai.twice[them] & ~ai.twice[c]);
    if (popcount(ai.pieces[them][4]) == 1) {
        const chess::Square q(lsb(ai.pieces[them][4]));
        const U64 target = ~ai.occ[c] & ~strongly;
        f(4, popcount(chess::attacks::knight(q).getBits() & ai.by[c][1] & target));
        const U64 sl = (chess::attacks::bishop(q, chess::Bitboard(occ)).getBits() & ai.by[c][2]) |
                       (chess::attacks::rook(q, chess::Bitboard(occ)).getBits() & ai.by[c][3]);
        f(5, popcount(sl & target & ai.twice[c]));
    }
    // Restricted: squares both sides attack that the enemy doesn't hold strongly.
    f(6, popcount(ai.all[them] & ai.all[c] & ~strongly));
}

template <typename F>
inline void pieces2(const AttackInfo& ai, int c, F&& f) {
    using namespace bb;
    const bool white = (c == 0);
    const int them = c ^ 1;
    const U64 occ = ai.occ[0] | ai.occ[1];
    const U64 pawns = ai.pieces[0][0] | ai.pieces[1][0];
    const int ksq = lsb(ai.pieces[c][5]);
    const U64 center = (1ULL << 27) | (1ULL << 28) | (1ULL << 35) | (1ULL << 36);  // d4 e4 d5 e5
    for (int t = 1; t <= 2; ++t)
        for (U64 p = ai.pieces[c][t]; p; p &= p - 1) {
            const int sq = lsb(p);
            const int front = white ? sq + 8 : sq - 8;
            if (front >= 0 && front < 64 && (pawns & (1ULL << front))) f(0, 1);
            f(t == 1 ? 2 : 3, distance(sq, ksq));
            if (t == 2 && popcount(chess::attacks::bishop(chess::Square(sq), chess::Bitboard(pawns)).getBits() & center) >= 2)
                f(1, 1);
        }
    const U64 queens = ai.pieces[0][4] | ai.pieces[1][4];
    const U64 excluded = ai.pieces[c][0] | pawn_attacks(ai.pieces[them][0], !white);
    for (U64 p = ai.pieces[c][3]; p; p &= p - 1) {
        const int sq = lsb(p);
        if (file_mask(sq % 8) & queens) f(6, 1);
        const int mob = popcount(chess::attacks::rook(chess::Square(sq), chess::Bitboard(occ)).getBits() & ~excluded);
        const int kf = ksq % 8, rf = sq % 8;
        if (mob <= 3 && rel_rank(ksq, white) == 0 && rel_rank(sq, white) == 0 && ((kf < 4) == (rf < kf))) {
            f(4, 1);
            if (!ai.castle[c][0] && !ai.castle[c][1]) f(5, 1);
        }
    }
    // Weak queen: an enemy bishop / rook lined up with it with one piece between.
    for (U64 q = ai.pieces[c][4]; q; q &= q - 1) {
        const int qsq = lsb(q);
        const chess::Square qs(qsq);
        U64 snipers = (chess::attacks::bishop(qs, chess::Bitboard(0)).getBits() & ai.pieces[them][2]) |
                      (chess::attacks::rook(qs, chess::Bitboard(0)).getBits() & ai.pieces[them][3]);
        int weak = 0;
        for (; snipers; snipers &= snipers - 1) {
            const U64 between = between_table().t[qsq][lsb(snipers)] & occ;
            if (between && (between & (between - 1)) == 0) weak = 1;
        }
        if (weak) f(7, 1);
    }
}

inline void imbalance2_counts(const AttackInfo& ai, int c, int cnt[6]) {
    using namespace bb;
    const U64 b = ai.pieces[c][2];
    cnt[0] = ((b & DarkSquares) && (b & ~DarkSquares)) ? 1 : 0;
    for (int t = 0; t <= 4; ++t) cnt[t + 1] = popcount(ai.pieces[c][t]);
}
template <typename F>
inline void imbalance2(const AttackInfo& ai, int c, F&& f) {
    int ours[6], theirs[6];
    imbalance2_counts(ai, c, ours);
    imbalance2_counts(ai, c ^ 1, theirs);
    int k = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j, ++k)
            if (ours[i] && ours[j]) f(k, ours[i] * ours[j]);
    for (int i = 1; i < 6; ++i)
        for (int j = 0; j < i; ++j, ++k)
            if (ours[i] && theirs[j]) f(k, ours[i] * theirs[j]);
}

template <typename F>
inline void passed2(const AttackInfo& ai, int c, F&& f) {
    using namespace bb;
    const bool white = (c == 0);
    const int them = c ^ 1;
    for (U64 p = passed_pawns(ai, c); p; p &= p - 1) {
        const int sq = lsb(p);
        const int r = rel_rank(sq, white), file = sq % 8;
        f(24, std::min(file, 7 - file));
        if (r < 3) continue;
        const U64 path = M.forward_file[c][sq];
        const U64 unsafe = path & (ai.all[them] | ai.occ[them]);
        const int stop = white ? sq + 8 : sq - 8;
        if (!unsafe) f(r, 1);
        else if (!(unsafe & (1ULL << stop))) f(8 + r, 1);
        if ((path & ~ai.all[c]) == 0) f(16 + r, 1);
    }
}

// ---- Pawn-only terms --------------------------------------------------------
// Everything in side c's evaluation that depends only on the two pawn
// bitboards: structure, the pawn-only part of passed / candidate pawns, space.
struct PawnSide {
    MgEg structure{};  // TermStructure
    MgEg passed{};     // TermPassed: per-rank bonus, protected, outside, candidates
    int space = 0;     // TermSpace (mg)
    bb::U64 passers = 0;
};

inline PawnSide pawn_side(const AttackInfo& ai, int c) {
    using namespace bb;
    PawnSide ps;
    auto add = [](MgEg& t, const MgEg& v, int n = 1) {
        t.mg += v.mg * n;
        t.eg += v.eg * n;
    };
    const bool white = (c == 0);
    const int them = c ^ 1;
    const U64 ours_p = ai.pieces[c][0], theirs_p = ai.pieces[them][0];
    const U64 our_attacks = pawn_attacks(ours_p, white), their_attacks = pawn_attacks(theirs_p, !white);
    U64 pawns = ours_p;
    while (pawns) {
        const int sq = lsb(pawns);
        pawns &= pawns - 1;
        const int f = sq % 8, r = rel_rank(sq, white);
        const U64 bit = 1ULL << sq;

        if (EvalUsePawnStructure) {
            const bool isolated = (adjacent_files(f) & ours_p) == 0;
            if (isolated) add(ps.structure, IsolatedPawn);
            if (M.forward_file[c][sq] & ours_p) add(ps.structure, DoubledPawn);  // another own pawn ahead
            if (!isolated) {
                // Backward: no own pawn on an adjacent file level or behind, and
                // the stop square is attacked by an enemy pawn.
                const U64 behind_or_level = ~ranks_ahead(sq / 8, white);
                const int stop = white ? sq + 8 : sq - 8;
                if ((adjacent_files(f) & behind_or_level & ours_p) == 0 && stop >= 0 && stop < 64 &&
                    (their_attacks & (1ULL << stop)))
                    add(ps.structure, BackwardPawn);
            }
            if (EvalUseConnected)
                pawn_links(ai, c, sq, [&](int kind, int i, int n) {
                    switch (kind) {
                        case 0: ps.structure.mg += PhalanxMG[i]; ps.structure.eg += PhalanxEG[i]; break;
                        case 1: ps.structure.mg += SupportedMG[i]; ps.structure.eg += SupportedEG[i]; break;
                        case 2: add(ps.structure, SupportCount, n); break;
                        case 3: add(ps.structure, ConnectedOpposed); break;
                        case 4: add(ps.structure, WeakUnopposed); break;
                        default: add(ps.structure, WeakLever); break;
                    }
                });
        }

        const bool passed = (M.passed[c][sq] & theirs_p) == 0 && (M.forward_file[c][sq] & ours_p) == 0;
        if (passed) ps.passers |= bit;
        if (EvalUsePassed && passed) {
            ps.passed.mg += PassedMG[r];
            ps.passed.eg += PassedEG[r];
            if (our_attacks & bit) add(ps.passed, PassedProtected);
            if (EvalUsePassedExtra && (f <= 1 || f >= 6)) {
                // Outside passer: on a wing, away from the enemy pawns, so it
                // drags the enemy king away from everything else.
                U64 near = 0;
                for (int df = -1; df <= 1; ++df)
                    if (f + df >= 0 && f + df <= 7) near |= file_mask(f + df);
                if (theirs_p && (theirs_p & near) == 0) ps.passed.eg += PassedOutsideEG;
            }
        } else if (EvalUseCandidate && (M.forward_file[c][sq] & (ours_p | theirs_p)) == 0) {
            // Candidate: half-open file, and at least as many own pawns able to
            // support its advance as enemy pawns guarding its path.
            const int sentries = popcount(M.attack_span[c][sq] & theirs_p);
            const int helpers = popcount(adjacent_files(f) & ~ranks_ahead(sq / 8, white) & ours_p);
            if (helpers >= sentries) ps.passed.eg += CandidateEG[r];
        }
    }

    // Space: safe central squares in our half (files c-f, ranks 2-4), squares
    // behind our own pawns counting double.
    if (EvalUseSpace) {
        const U64 area = (file_mask(2) | file_mask(3) | file_mask(4) | file_mask(5)) &
                         (white ? (rank_mask(1) | rank_mask(2) | rank_mask(3))
                                : (rank_mask(6) | rank_mask(5) | rank_mask(4)));
        const U64 safe = area & ~ours_p & ~their_attacks;
        U64 behind = ours_p;
        for (int i = 0; i < 3; ++i) behind |= white ? (behind >> 8) : (behind << 8);
        ps.space = SpacePerSquare * (popcount(safe) + popcount(safe & behind));
    }
    return ps;
}

// Pawn hash (SP_PAWN_HASH): pawn_side() for both colours, keyed by the two
// pawn bitboards themselves (stored whole, so there are no false hits).
// One table per thread.
#ifndef SP_PAWN_HASH
#define SP_PAWN_HASH 1
#endif
inline constexpr bool UsePawnHash = SP_PAWN_HASH;

class PawnHash {
public:
    static constexpr std::size_t Size = 1 << 13;  // entries (~0.6 MB)
    const PawnSide* probe(const AttackInfo& ai) {
        const bb::U64 wp = ai.pieces[0][0], bp = ai.pieces[1][0];
        Entry& e = table_[index(wp, bp)];
        if (!e.used || e.wp != wp || e.bp != bp) {
            e.wp = wp;
            e.bp = bp;
            e.used = true;
            e.side[0] = pawn_side(ai, 0);
            e.side[1] = pawn_side(ai, 1);
        }
        return e.side;
    }
    static PawnHash& local() {
        static thread_local PawnHash h;
        return h;
    }

private:
    struct Entry {
        bb::U64 wp = 0, bp = 0;
        PawnSide side[2];
        bool used = false;
    };
    static std::size_t index(bb::U64 wp, bb::U64 bp) {
        const bb::U64 x = (wp * 0x9E3779B97F4A7C15ULL) ^ (bp * 0xC2B2AE3D27D4EB4FULL);
        return static_cast<std::size_t>(x >> 51) & (Size - 1);
    }
    std::vector<Entry> table_ = std::vector<Entry>(Size);
};

// ps: side c's pawn-only terms, if already known (pawn hash); nullptr = compute.
inline TermScores side_positional(const AttackInfo& ai, int c, const PawnSide* ps = nullptr) {
    using namespace bb;
    TermScores ts;
    auto add = [&](Term term, const MgEg& v, int n = 1) {
        ts.t[term].mg += v.mg * n;
        ts.t[term].eg += v.eg * n;
    };
    const bool white = (c == 0);
    const int them = c ^ 1;
    const U64 ours_p = ai.pieces[c][0], theirs_p = ai.pieces[them][0];
    const U64 occ = ai.occ[0] | ai.occ[1];
    const int our_k = lsb(ai.pieces[c][5]), their_k = lsb(ai.pieces[them][5]);

    // Pawns: structure and passed pawns. The pawn-only part comes from
    // pawn_side() (cached by the pawn hash when SP_PAWN_HASH is on); the
    // terms that also depend on pieces and kings are added per passer here.
    const PawnSide own_ps = ps ? PawnSide{} : pawn_side(ai, c);
    const PawnSide& pw = ps ? *ps : own_ps;
    ts.t[TermStructure].mg += pw.structure.mg;
    ts.t[TermStructure].eg += pw.structure.eg;
    ts.t[TermPassed].mg += pw.passed.mg;
    ts.t[TermPassed].eg += pw.passed.eg;
    bool unstoppable = false;  // at least one passer the enemy king can't catch
    for (U64 pp = EvalUsePassed ? pw.passers : 0; pp; pp &= pp - 1) {
        const int sq = lsb(pp);
        const int f = sq % 8, r = rel_rank(sq, white);
        const U64 bit = 1ULL << sq;
        const int stop = white ? sq + 8 : sq - 8;
        if (stop >= 0 && stop < 64) {
            if (occ & (1ULL << stop)) add(TermPassed, PassedBlocked);
            if ((M.forward_file[c][sq] & occ) == 0) ts.t[TermPassed].eg += PassedFreePathEG[r];
            if (r >= 3)
                ts.t[TermPassed].eg += PassedKingDistEG * (r - 2) * (distance(their_k, stop) * 2 - distance(our_k, stop));
        }
        if (EvalUseRooks) {
            const U64 behind = file_mask(f) & ~M.forward_file[c][sq] & ~bit;
            if (behind & ai.pieces[c][3]) add(TermRooks, RookBehindPasser);
        }
        if (EvalUsePassedExtra) {
            // Unstoppable (rule of the square) when the opponent has no pieces.
            const bool them_no_pieces =
                (ai.pieces[them][1] | ai.pieces[them][2] | ai.pieces[them][3] | ai.pieces[them][4]) == 0;
            if (them_no_pieces && (M.forward_file[c][sq] & occ) == 0) {
                const int promo = white ? 56 + f : f;
                const int moves = std::min(7 - r, 5);
                const int kd = distance(their_k, promo) - (ai.stm == them ? 1 : 0);
                if (kd > moves) unstoppable = true;
            }
        }
    }
    if (unstoppable) ts.t[TermPassed].eg += UnstoppableEG;

    // Knights / bishops: outposts; bishop pair / bad bishop.
    if (EvalUseOutposts) {
        for (int t = 1; t <= 2; ++t) {
            U64 pcs = ai.pieces[c][t];
            while (pcs) {
                const int sq = lsb(pcs);
                pcs &= pcs - 1;
                const int r = rel_rank(sq, white);
                if (r >= 3 && r <= 5 && (ai.by[c][0] & (1ULL << sq)) && (M.attack_span[c][sq] & theirs_p) == 0)
                    add(TermOutposts, t == 1 ? KnightOutpost : BishopOutpost);
            }
        }
    }
    if (EvalUseBishops) {
        const U64 bishops = ai.pieces[c][2];
        if ((bishops & DarkSquares) && (bishops & ~DarkSquares)) add(TermBishops, BishopPair);
        U64 pcs = bishops;
        while (pcs) {
            const int sq = lsb(pcs);
            pcs &= pcs - 1;
            const U64 colour = (DarkSquares >> sq) & 1 ? DarkSquares : ~DarkSquares;
            add(TermBishops, BadBishopPerPawn, popcount(ours_p & colour));
            if (EvalUseBadBishop3) {
                // Own pawns on the bishop's colour that are blocked: they won't leave it.
                const U64 fixed = ours_p & colour & (white ? (occ >> 8) : (occ << 8));
                add(TermBishops, BadBishopFixedPawn, popcount(fixed));
            }
        }
    }

    // Rooks: open files, 7th rank.
    if (EvalUseRooks) {
        U64 pcs = ai.pieces[c][3];
        while (pcs) {
            const int sq = lsb(pcs);
            pcs &= pcs - 1;
            const U64 file = file_mask(sq % 8);
            if ((file & ours_p) == 0) add(TermRooks, (file & theirs_p) ? RookSemiOpenFile : RookOpenFile);
            if (rel_rank(sq, white) == 6 &&
                (rel_rank(their_k, white) == 7 || (theirs_p & rank_mask(white ? 6 : 1))))
                add(TermRooks, RookOnSeventh);
        }
    }

    // Threats against the opponent's pieces.
    if (EvalUseThreats) {
        const U64 minors_majors = ai.pieces[them][1] | ai.pieces[them][2] | ai.pieces[them][3] | ai.pieces[them][4];
        add(TermThreats, ThreatByPawn, popcount(ai.by[c][0] & minors_majors));
        add(TermThreats, ThreatByMinor, popcount((ai.by[c][1] | ai.by[c][2]) & (ai.pieces[them][3] | ai.pieces[them][4])));
        add(TermThreats, ThreatByRook, popcount(ai.by[c][3] & ai.pieces[them][4]));
        add(TermThreats, HangingPiece, popcount(ai.all[c] & minors_majors & ~ai.all[them]));
    }

    // Rooks and the opponent's passed pawns: behind them is active (Tarrasch),
    // in front of them is a passive blockade.
    if (EvalUseRookPassers && ai.pieces[c][3]) {
        U64 ep = passed_pawns(ai, them);
        while (ep) {
            const int sq = lsb(ep);
            ep &= ep - 1;
            const U64 file = file_mask(sq % 8);
            const U64 ahead = M.forward_file[them][sq];                 // towards its promotion square
            const U64 behind = file & ~ahead & ~(1ULL << sq);
            if (behind & ai.pieces[c][3]) add(TermRooks, RookBehindEnemyPasser);
            if (ahead & ai.pieces[c][3]) add(TermRooks, RookInFrontOfEnemyPasser);
        }
    }

    // Attacks on undefended enemy pawns.
    if (EvalUsePawnThreats) {
        const U64 loose = theirs_p & ~ai.all[them];
        add(TermThreats, ThreatOnPawn, popcount((ai.by[c][1] | ai.by[c][2] | ai.by[c][3]) & loose));
        ts.t[TermThreats].eg += KingThreatOnPawnEG * popcount(ai.by[c][5] & loose);
    }

    // Pawns that no enemy pawn defends, attacked by our minors / rooks.
    if (EvalUseWeakPawnThreats) {
        const U64 weak = theirs_p & ~ai.by[them][0];
        add(TermThreats, ThreatOnWeakPawn, popcount((ai.by[c][1] | ai.by[c][2] | ai.by[c][3]) & weak));
    }

    // Material imbalance: knights gain and rooks lose value with more own pawns.
    if (EvalUseImbalance) {
        const int extra = popcount(ours_p) - 5;
        add(TermBishops, KnightPerPawn, popcount(ai.pieces[c][1]) * extra);
        add(TermRooks, RookPerPawn, popcount(ai.pieces[c][3]) * extra);
    }

    if (EvalUseThreats2) threats2(ai, c, [&](int i, int n) { add(TermThreats, Threats2W[i], n); });
    if (EvalUsePieces2) pieces2(ai, c, [&](int i, int n) { add(TermOutposts, Pieces2W[i], n); });
    if (EvalUseImbalance2) imbalance2(ai, c, [&](int i, int n) { add(TermBishops, Imbalance2W[i], n); });
    if (EvalUsePassed2) passed2(ai, c, [&](int i, int n) { add(TermPassed, Passed2W[i], n); });

    // King safety of OUR king.
    if (EvalUseKingSafety) {
        if (EvalUseDanger2) {
            const MgEg d = danger2_penalty(ai, c);
            ts.t[TermKingSafety].mg -= d.mg;
            ts.t[TermKingSafety].eg -= d.eg;
        } else {
            ts.t[TermKingSafety].mg -= danger_penalty(ai, c);
        }

        if (EvalUseShelter) {
            ts.t[TermKingSafety].mg += shelter_at(ai, c, shelter_square(ai, c));
            ts.t[TermKingSafety].eg += KingPawnDistEG * king_pawn_distance(ai, c);
        } else if (rel_rank(our_k, white) <= 1) {
            // Pawn shield, for a king on its first two ranks.
            const int kf = std::clamp(our_k % 8, 1, 6);
            for (int f = kf - 1; f <= kf + 1; ++f) {
                const U64 file = file_mask(f);
                const U64 own = file & ours_p;
                if (!own) ts.t[TermKingSafety].mg += ShieldMissing;
                else {
                    const int nearest = white ? lsb(own) : 63 - std::countl_zero(own);
                    if (rel_rank(nearest, white) >= 2) ts.t[TermKingSafety].mg += ShieldRank3;
                }
                if (!(file & (ours_p | theirs_p))) ts.t[TermKingSafety].mg += ShieldOpenFile;
            }
        }
    }

    if (EvalUseMobilityArea) {
        ts.t[TermMobilityArea].mg += ai.mobility_adjust[c].mg;
        ts.t[TermMobilityArea].eg += ai.mobility_adjust[c].eg;
    }

    // Space: safe central squares in our half (files c-f, ranks 2-4), squares
    // behind our own pawns counting double.
    if (EvalUseSpace) ts.t[TermSpace].mg += pw.space;
    return ts;
}

// ---- KPK table ---------------------------------------------------------------
// King + pawn vs king, solved once by retrograde analysis. Positions are
// stored with the pawn side as White and the pawn on files a-d (others are
// mirrored). kpk_win(): true = White (the pawn side) wins with best play.
class Kpk {
public:
    static const Kpk& get() {
        static const Kpk table;
        return table;
    }
    // Squares 0..63 with a1 = 0; pawn side White; stm 0 = White to move.
    bool win(int stm, int wk, int bk, int wp) const {
        if (wp % 8 > 3) { wk ^= 7; bk ^= 7; wp ^= 7; }  // mirror files
        return res_[index(stm, wk, bk, wp)] == Win;
    }

private:
    enum : std::uint8_t { Unknown = 0, Invalid, Draw, Win };
    std::vector<std::uint8_t> res_ = std::vector<std::uint8_t>(2 * 64 * 64 * 24, Unknown);

    static int index(int stm, int wk, int bk, int wp) {
        const int p = (wp / 8 - 1) * 4 + (wp % 8);  // rank 2..7, file a..d
        return ((stm * 64 + wk) * 64 + bk) * 24 + p;
    }
    static int dist(int a, int b) { return std::max(std::abs(a % 8 - b % 8), std::abs(a / 8 - b / 8)); }
    static bool pawn_attacks_sq(int wp, int sq) {
        return sq / 8 == wp / 8 + 1 && std::abs(sq % 8 - wp % 8) == 1;
    }

    Kpk() {
        // Initial classification.
        for (int stm = 0; stm < 2; ++stm)
            for (int wk = 0; wk < 64; ++wk)
                for (int bk = 0; bk < 64; ++bk)
                    for (int r = 1; r <= 6; ++r)
                        for (int f = 0; f < 4; ++f) {
                            const int wp = r * 8 + f;
                            std::uint8_t& v = res_[index(stm, wk, bk, wp)];
                            if (wk == bk || wk == wp || bk == wp || dist(wk, bk) <= 1 ||
                                (stm == 0 && pawn_attacks_sq(wp, bk)))
                                v = Invalid;  // overlapping, kings touching, or the side not to move in check
                            else if (stm == 0 && r == 6 && wk != wp + 8 && bk != wp + 8 &&
                                     (dist(bk, wp + 8) > 1 || dist(wk, wp + 8) == 1))
                                v = Win;  // promotes safely
                            else if (stm == 1 && ((dist(bk, wp) == 1 && dist(wk, wp) > 1) || black_stalemated(wk, bk, wp)))
                                v = Draw;  // pawn falls, or stalemate
                        }
        // Iterate until nothing changes.
        for (bool changed = true; changed;) {
            changed = false;
            for (int stm = 0; stm < 2; ++stm)
                for (int wk = 0; wk < 64; ++wk)
                    for (int bk = 0; bk < 64; ++bk)
                        for (int r = 1; r <= 6; ++r)
                            for (int f = 0; f < 4; ++f) {
                                const int wp = r * 8 + f;
                                std::uint8_t& v = res_[index(stm, wk, bk, wp)];
                                if (v != Unknown) continue;
                                const std::uint8_t nv = stm == 0 ? white_moves(wk, bk, wp) : black_moves(wk, bk, wp);
                                if (nv != Unknown) { v = nv; changed = true; }
                            }
        }
        for (auto& v : res_)
            if (v == Unknown) v = Draw;
    }

    bool black_stalemated(int wk, int bk, int wp) const {
        for (int d = 0; d < 64; ++d) {
            if (dist(bk, d) != 1) continue;
            if (dist(wk, d) <= 1 || pawn_attacks_sq(wp, d)) continue;
            return false;
        }
        return !pawn_attacks_sq(wp, bk);
    }
    std::uint8_t lookup(int stm, int wk, int bk, int wp) const { return res_[index(stm, wk, bk, wp)]; }

    // White to move: a win if some move reaches a win; a draw once every move is a draw.
    std::uint8_t white_moves(int wk, int bk, int wp) const {
        bool all_draw = true;
        for (int d = 0; d < 64; ++d) {
            if (dist(wk, d) != 1 || d == wp || dist(d, bk) <= 1) continue;
            const std::uint8_t v = lookup(1, d, bk, wp);
            if (v == Win) return Win;
            if (v != Draw && v != Invalid) all_draw = false;
        }
        const int up = wp + 8;
        if (wp / 8 < 6 && up != wk && up != bk) {
            const std::uint8_t v = lookup(1, wk, bk, up);
            if (v == Win) return Win;
            if (v != Draw && v != Invalid) all_draw = false;
            if (wp / 8 == 1 && up + 8 != wk && up + 8 != bk) {
                const std::uint8_t v2 = lookup(1, wk, bk, up + 8);
                if (v2 == Win) return Win;
                if (v2 != Draw && v2 != Invalid) all_draw = false;
            }
        }
        return all_draw ? Draw : Unknown;
    }
    // Black to move: a draw if some move reaches a draw (or takes the pawn safely); a win once every move loses.
    std::uint8_t black_moves(int wk, int bk, int wp) const {
        bool all_win = true, any = false;
        for (int d = 0; d < 64; ++d) {
            if (dist(bk, d) != 1 || dist(d, wk) <= 1 || pawn_attacks_sq(wp, d)) continue;
            if (d == wp) return Draw;  // undefended pawn taken (defended case excluded by dist to wk)
            any = true;
            const std::uint8_t v = lookup(0, wk, d, wp);
            if (v == Draw) return Draw;
            if (v != Win) all_win = false;
        }
        if (!any) return Draw;  // stalemate
        return all_win ? Win : Unknown;
    }
};

// Endgame scale factor (64 = normal) applied to the endgame score of the
// side that is ahead: opposite-coloured bishops, and material that can't win.
inline int endgame_scale(const AttackInfo& ai, Score eg_white_view) {
    using namespace bb;
    if (!EvalUseScaling) return 64;
    const int strong = eg_white_view >= 0 ? 0 : 1, weak = strong ^ 1;
    auto npm = [&](int c) {
        return popcount(ai.pieces[c][1]) * PieceValueMG[1] + popcount(ai.pieces[c][2]) * PieceValueMG[2] +
               popcount(ai.pieces[c][3]) * PieceValueMG[3] + popcount(ai.pieces[c][4]) * PieceValueMG[4];
    };
    // Strong side has no pawns and at most a minor piece more: hard or impossible to win.
    if (ai.pieces[strong][0] == 0 && npm(strong) - npm(weak) <= PieceValueMG[2])
        return npm(strong) < PieceValueMG[3] ? 0 : 16;
    // Opposite-coloured bishops.
    const U64 wb = ai.pieces[0][2], bbish = ai.pieces[1][2];
    if (popcount(wb) == 1 && popcount(bbish) == 1 &&
        (((wb & DarkSquares) != 0) != ((bbish & DarkSquares) != 0))) {
        const bool only_bishops = !(ai.pieces[0][1] | ai.pieces[1][1] | ai.pieces[0][3] | ai.pieces[1][3] |
                                    ai.pieces[0][4] | ai.pieces[1][4]);
        return only_bishops ? 32 : 48;
    }
    return 64;
}

// ---- Initiative (EV_INITIATIVE) ----------------------------------------------
enum InitFeature {
    InPawns,             // pawns on the board
    InPassed,            // passed pawns, both sides
    InBothFlanks,        // pawns on both wings (files a-d and e-h) (0/1)
    InOutflanking,       // file distance between the kings - rank distance
    InInfiltration,      // a king on the opponent's half (0/1)
    InPawnsOnly,         // no pieces left, pawns only (0/1)
    InAlmostUnwinnable,  // pawns on one wing and the kings not outflanking (0/1)
    InBias,              // always 1
    InCount
};
static_assert(InCount == 8);
struct InitSignals {
    int x[InCount]{};
};

inline InitSignals initiative_signals(const AttackInfo& ai) {
    using namespace bb;
    InitSignals k;
    const U64 pawns = ai.pieces[0][0] | ai.pieces[1][0];
    const int wk = lsb(ai.pieces[0][5]), bk = lsb(ai.pieces[1][5]);
    const U64 queenside = file_mask(0) | file_mask(1) | file_mask(2) | file_mask(3);
    k.x[InPawns] = popcount(pawns);
    k.x[InPassed] = popcount(passed_pawns(ai, 0)) + popcount(passed_pawns(ai, 1));
    k.x[InBothFlanks] = (pawns & queenside) && (pawns & ~queenside) ? 1 : 0;
    k.x[InOutflanking] = std::abs(wk % 8 - bk % 8) - std::abs(wk / 8 - bk / 8);
    k.x[InInfiltration] = (wk / 8 >= 4 || bk / 8 <= 3) ? 1 : 0;
    bool pieces = false;
    for (int c = 0; c < 2; ++c)
        for (int t = 1; t <= 4; ++t) pieces |= ai.pieces[c][t] != 0;
    k.x[InPawnsOnly] = pieces ? 0 : 1;
    k.x[InAlmostUnwinnable] = (k.x[InOutflanking] < 0 && !k.x[InBothFlanks]) ? 1 : 0;
    k.x[InBias] = 1;
    return k;
}
inline int initiative_value(const InitSignals& k) {
    int c = 0;
    for (int i = 0; i < InCount; ++i) c += InitWeight[i] * k.x[i];
    return c;
}
inline Score apply_initiative(Score eg, int c) {
    if (eg == 0) return eg;
    const int sgn = eg > 0 ? 1 : -1;
    return eg + sgn * std::max(c, -std::abs(eg));
}

// ---- Endgame scale factors (EV_SCALE2) --------------------------------------
enum ScaleCat : std::uint8_t {
    ScFewPawns, ScFewPawnsOcb, ScOcbOnly, ScQueenVsMinors, ScNoPawns, ScZero, ScKrkp, ScKqkp, ScNormal
};
struct ScaleInfo {
    std::uint8_t cat = ScNormal;
    std::uint8_t cnt = 0;
};

// Which scale factor applies when side `strong` is ahead.
inline ScaleInfo scale_info(const AttackInfo& ai, int strong) {
    using namespace bb;
    const int weak = strong ^ 1;
    auto npm = [&](int c) {
        return popcount(ai.pieces[c][1]) * PieceValueMG[1] + popcount(ai.pieces[c][2]) * PieceValueMG[2] +
               popcount(ai.pieces[c][3]) * PieceValueMG[3] + popcount(ai.pieces[c][4]) * PieceValueMG[4];
    };
    const int npm_s = npm(strong), npm_w = npm(weak);
    const int pawns_s = popcount(ai.pieces[strong][0]), pawns_w = popcount(ai.pieces[weak][0]);
    const int ks = lsb(ai.pieces[strong][5]), kw = lsb(ai.pieces[weak][5]);
    const bool strong_white = strong == 0;

    // KPK: exact, from the table.
    if (npm_s == 0 && npm_w == 0 && pawns_s + pawns_w == 1) {
        const int pside = pawns_s ? strong : weak;
        int wk = lsb(ai.pieces[pside][5]), bk = lsb(ai.pieces[pside ^ 1][5]), wp = lsb(ai.pieces[pside][0]);
        if (pside == 1) { wk ^= 56; bk ^= 56; wp ^= 56; }
        const bool win = Kpk::get().win(ai.stm == pside ? 0 : 1, wk, bk, wp);
        return {static_cast<std::uint8_t>(win && pside == strong ? ScNormal : ScZero), 0};
    }
    // Bishop and rook pawns of the wrong colour vs a king in the corner.
    if (npm_s == PieceValueMG[2] && ai.pieces[strong][2] && pawns_s && npm_w == 0) {
        const U64 ps = ai.pieces[strong][0];
        for (int f : {0, 7})
            if ((ps & ~file_mask(f)) == 0) {
                const int q = f + (strong_white ? 56 : 0);
                const bool bishop_dark = (ai.pieces[strong][2] & DarkSquares) != 0;
                const bool q_dark = ((DarkSquares >> q) & 1) != 0;
                if (bishop_dark != q_dark && distance(kw, q) <= 1) return {ScZero, 0};
            }
    }
    // KRKP / KQKP with the pawn far advanced and supported by its king.
    if (pawns_s == 0 && npm_w == 0 && pawns_w == 1 &&
        ((npm_s == PieceValueMG[3] && ai.pieces[strong][3]) || (npm_s == PieceValueMG[4] && ai.pieces[strong][4]))) {
        const int pw = lsb(ai.pieces[weak][0]);
        const int r = rel_rank(pw, !strong_white);  // from the pawn's side
        const int qsq = pw % 8 + (strong_white ? 0 : 56);
        if (ai.pieces[strong][3]) {
            const bool in_front = (M.forward_file[weak][pw] & (1ULL << ks)) != 0;
            if (r >= 4 && distance(kw, pw) <= 1 && distance(ks, qsq) >= 3 && !in_front) return {ScKrkp, 0};
        } else {
            const int f = pw % 8;
            if (r == 6 && (f == 0 || f == 2 || f == 5 || f == 7) && distance(kw, pw) <= 1 && distance(ks, pw) >= 3)
                return {ScKqkp, 0};
        }
    }
    // No pawns and at most a minor piece ahead.
    if (pawns_s == 0 && npm_s - npm_w <= PieceValueMG[2])
        return {static_cast<std::uint8_t>(npm_s < PieceValueMG[3] ? ScZero : ScNoPawns), 0};
    // Opposite-coloured bishops.
    const U64 wb = ai.pieces[0][2], blb = ai.pieces[1][2];
    const bool ocb = popcount(wb) == 1 && popcount(blb) == 1 && (((wb & DarkSquares) != 0) != ((blb & DarkSquares) != 0));
    if (ocb && npm_s == PieceValueMG[2] && npm_w == PieceValueMG[2]) {
        const int passed = popcount(passed_pawns(ai, 0)) + popcount(passed_pawns(ai, 1));
        return {ScOcbOnly, static_cast<std::uint8_t>(passed)};
    }
    // One queen against minor pieces only.
    const int queens = popcount(ai.pieces[0][4]) + popcount(ai.pieces[1][4]);
    if (queens == 1) {
        const int qside = ai.pieces[0][4] ? 0 : 1, other = qside ^ 1;
        const int minors = popcount(ai.pieces[other][1] | ai.pieces[other][2]);
        if (minors && !ai.pieces[other][3])
            return {ScQueenVsMinors, static_cast<std::uint8_t>(minors)};
    }
    return {static_cast<std::uint8_t>(ocb ? ScFewPawnsOcb : ScFewPawns), static_cast<std::uint8_t>(pawns_s)};
}

// Scale factor (0..64) for a category with the given parameters (engine ints or tuner doubles).
template <typename T>
inline T scale_from(const ScaleInfo& si, const T* p) {
    T v = 64;
    switch (si.cat) {
        case ScFewPawns: v = std::min<T>(64, p[0] + p[1] * si.cnt); break;
        case ScFewPawnsOcb: v = std::min<T>(64, p[0] + p[2] * si.cnt); break;
        case ScOcbOnly: v = std::min<T>(64, p[3] + p[4] * si.cnt); break;
        case ScQueenVsMinors: v = std::min<T>(64, p[5] + p[6] * si.cnt); break;
        case ScNoPawns: v = p[7]; break;
        case ScZero: v = 0; break;
        case ScKrkp: v = p[8]; break;
        case ScKqkp: v = p[9]; break;
        default: break;
    }
    return std::max<T>(0, v);
}

// All positional terms, white minus black, plus the endgame scale factor.
struct Positional {
    MgEg score;              // white minus black
    TermScores side[2];      // per side, own point of view (for display)
    AttackInfo ai;
};
inline Positional positional(const chess::Board& b) {
    Positional p;
    p.ai = gather_attacks(b);
    const PawnSide* ps = UsePawnHash ? PawnHash::local().probe(p.ai) : nullptr;
    p.side[0] = side_positional(p.ai, 0, ps ? &ps[0] : nullptr);
    p.side[1] = side_positional(p.ai, 1, ps ? &ps[1] : nullptr);
    const MgEg w = p.side[0].total(), bl = p.side[1].total();
    p.score = {w.mg - bl.mg, w.eg - bl.eg};
    return p;
}

// One side's middlegame / endgame totals (its own point of view).
inline MgEg side_totals(const SideTerms& s, const MgEg& slider) {
    return {s.material_mg + s.pst_mg + s.mobility_mg + slider.mg,
            s.material_eg + s.pst_eg + s.mobility_eg + slider.eg};
}

// Fifty-move scaling: the score shrinks linearly towards 0 as the halfmove
// clock approaches 100 (a draw by rule), so the engine prefers progress
// (captures, pawn moves) over shuffling when it is better.
// Known-ending scores (KnownWin and above, see known_ending) are left alone:
// at 10000 cp the scaling moved them by 100 cp per ply, five times the
// mop-up terms, and the search stopped making progress towards mate.
inline constexpr Score KnownWin = 10000;
inline Score fifty_move_scale(Score s, int halfmove_clock) {
    if (std::abs(s) >= KnownWin - 1000) return s;
    return s * (100 - std::min(halfmove_clock, 100)) / 100;
}

// White's point of view, before fifty-move scaling (which depends on the
// halfmove clock, not part of the position key: the TT caches this value).
inline Score white_view_unscaled(const chess::Board& board, const Terms& t) {
    const Positional p = positional(board);
    const MgEg w = side_totals(t.side[0], p.ai.slider[0]);
    const MgEg b = side_totals(t.side[1], p.ai.slider[1]);
    const Score mg = w.mg - b.mg + p.score.mg;
    Score eg = w.eg - b.eg + p.score.eg;
    if (EvalUseInitiative) eg = apply_initiative(eg, initiative_value(initiative_signals(p.ai)));
    if (EvalUseScale2) eg = eg * scale_from(scale_info(p.ai, eg > 0 ? 0 : 1), ScaleParam) / 64;
    else eg = eg * endgame_scale(p.ai, eg) / 64;
    Score s = taper(mg, eg, t.phase());
    if (EvalUseTempo) s += board.sideToMove() == chess::Color::WHITE ? Tempo : -Tempo;
    return s;
}

// White's point of view.
inline Score white_view(const chess::Board& board, const Terms& t) {
    return fifty_move_scale(white_view_unscaled(board, t), static_cast<int>(board.halfMoveClock()));
}

// ---------------------------------------------------------------------------
// Board with incrementally updated evaluation terms.
// ---------------------------------------------------------------------------
// NNUE (EV_NNUE=1): the network in Nnue.h / NetData.h replaces the
// hand-written evaluation in the search; EvalBoard then keeps the network's
// accumulators up to date instead of the incremental terms.
inline constexpr bool UseNnue = EV_NNUE;

class EvalBoard : public chess::Board {
public:
    explicit EvalBoard(std::string_view fen = chess::constants::STARTPOS) : chess::Board(fen) {
        refresh();
    }
    explicit EvalBoard(const chess::Board& board) : chess::Board(board) { refresh(); }

    void setFen(std::string_view fen) override {
#if EV_NNUE
        lazy_ = false;  // a new position: eager updates until lazy_begin()
#endif
        chess::Board::setFen(fen);
        refresh();
    }

    // Current incremental terms. No computation: just a read.
    const Terms& terms() const { return terms_; }

    // Repetition draw for the search (RepRule 1, Search.h): an earlier
    // occurrence inside the search tree (at most `ply` half-moves back) is a
    // draw at once, one in the game before the search started only as the
    // third occurrence. chess::Board::isRepetition(1) counts any earlier
    // occurrence, game history included.
    bool repetition_draw(int ply) const {
        const int size = static_cast<int>(prev_states_.size());
        int count = 0;
        for (int i = size - 2; i >= 0 && i >= size - hfm_ - 1; i -= 2) {
            if (prev_states_[static_cast<std::size_t>(i)].hash != key_) continue;
            if (size - i <= ply) return true;  // repeated within the search
            if (++count >= 2) return true;     // third occurrence in all
        }
        return false;
    }

    // Recompute from scratch (after setup; also handy for debugging).
    void refresh() {
        terms_ = compute(*this);
#if EV_NNUE
        acc_.refresh(*this, *nnue::network());
#endif
    }

#if EV_NNUE
    const nnue::Accumulator& accumulator() const { return acc_; }

    // Lazy accumulator (nnue::AccStack): on while a Searcher owns the board.
    // make() / unmake() must then be used instead of makeMove / unmakeMove.
    static constexpr bool LazyPossible = SP_LAZY_ACC && nnue::KingBuckets == 1;
    void lazy_begin() {
        if constexpr (LazyPossible) {
            stack_.start(acc_);
            lazy_ = true;
        }
    }
    bool lazy() const { return lazy_; }
    const std::int16_t* lazy_acc(int persp) const { return stack_.current(persp); }
#endif
    void make(const chess::Move& m) {
#if EV_NNUE
        if (lazy_) stack_.push();
#endif
        makeMove(m);
    }
    void unmake(const chess::Move& m) {
#if EV_NNUE
        if (lazy_) {
            replaying_ = true;  // the pieces move back; the frame is simply dropped
            unmakeMove(m);
            replaying_ = false;
            stack_.pop();
            return;
        }
#endif
        unmakeMove(m);
    }

protected:
    void placePiece(chess::Piece piece, chess::Square sq) override {
        chess::Board::placePiece(piece, sq);
#if EV_NNUE
        if (replaying_) return;
        if (lazy_) {
            stack_.record(*nnue::network(), static_cast<int>(piece.color().internal()), static_cast<int>(piece.type().internal()), sq.index(), true);
            return;
        }
        acc_.add(*nnue::network(), static_cast<int>(piece.color().internal()), static_cast<int>(piece.type().internal()), sq.index());
        if constexpr (nnue::KingBuckets > 1) {
            // A king that changes its side's bucket or mirroring: rebuild that side.
            if (piece.type() == chess::PieceType::KING) {
                const int c = static_cast<int>(piece.color().internal());
                if (nnue::king_config(c, sq.index()) != acc_.cfg[c]) acc_.refresh_side(*this, *nnue::network(), c, sq.index());
            }
        }
#else
        terms_.add(piece, sq);
#endif
    }

    void removePiece(chess::Piece piece, chess::Square sq) override {
#if EV_NNUE
        if (lazy_ || replaying_) {
            if (lazy_ && !replaying_)
                stack_.record(*nnue::network(), static_cast<int>(piece.color().internal()), static_cast<int>(piece.type().internal()), sq.index(), false);
            chess::Board::removePiece(piece, sq);
            return;
        }
        acc_.sub(*nnue::network(), static_cast<int>(piece.color().internal()), static_cast<int>(piece.type().internal()), sq.index());
#else
        terms_.sub(piece, sq);
#endif
        chess::Board::removePiece(piece, sq);
    }

private:
    Terms terms_;  // hand-written eval's incremental terms (not updated with EV_NNUE)
#if EV_NNUE
    nnue::Accumulator acc_;
    mutable nnue::AccStack stack_;
    bool lazy_ = false;
    bool replaying_ = false;
#endif
};

// ---------------------------------------------------------------------------
// Known endgames (EV_KXK). The search audit (Oct 2026) found the net badly
// wrong with little material: KQK +208, KQK with mate in one on the board
// +38, KRK +146, KNNK (a draw) +63; the engine could not mate KQK or KRK in
// 5 s. Datagen made it worse: games never reached the +1200 adjudication in
// these endings, the engine then failed to mate within 50 moves, and 80% of
// the KRK positions in a 300-game sample were labelled draws. These rules
// replace the net in the positions they cover:
//   * lone king against a queen, a rook, bishops on both colours, or bishop
//     and knight (pawns of the strong side allowed): KnownWin + material,
//     plus terms that drive the lone king to the edge (to a corner of the
//     bishop's colour for KBNK) and bring the kings together, so the search
//     makes progress towards mate;
//   * no pawns, one side a bare king, the other at most two knights or one
//     minor piece: 0 (no forced mate).
// g_kxk (KxkEval in tune builds) switches the rules off for A/B tests.
// The classification is in Endgames.h (shared with datagen and the trainer).
#ifndef EV_KXK
#define EV_KXK 1
#endif
inline int g_kxk = EV_KXK;

// Score of a known ending, White's point of view; false if none applies.
inline bool known_ending(const chess::Board& b, Score& white_score) {
    chess::Color strong;
    const KnownEnding kind = classify_ending(b, strong);
    if (kind == KnownEnding::None) return false;
    if (kind == KnownEnding::Draw) {
        white_score = 0;
        return true;
    }
    using chess::PieceType;
    const chess::Color weak = ~strong;
    const int q = b.pieces(PieceType::QUEEN, strong).count(), r = b.pieces(PieceType::ROOK, strong).count();
    const std::uint64_t bish = b.pieces(PieceType::BISHOP, strong).getBits();
    const int nb = std::popcount(bish), n = b.pieces(PieceType::KNIGHT, strong).count();
    const int p = b.pieces(PieceType::PAWN, strong).count();
    const int wk = b.kingSq(weak).index(), sk = b.kingSq(strong).index();
    const int wf = wk % 8, wr = wk / 8;
    const int dist = std::max(std::abs(wf - sk % 8), std::abs(wr - sk / 8));  // king distance 1..7
    int push;
    if (q == 0 && r == 0 && n > 0 && nb > 0 && !bishops_on_both_colours(bish)) {
        // KBNK: mate only in a corner of the bishop's colour (h1 / a8 are light).
        const bool light = (bish & LightSquares) != 0;
        const int c1 = light ? 7 : 0, c2 = light ? 56 : 63;
        auto cheb = [](int a, int c) { return std::max(std::abs(a % 8 - c % 8), std::abs(a / 8 - c / 8)); };
        push = 40 * (7 - std::min(cheb(wk, c1), cheb(wk, c2)));  // 280 = in the right corner
    } else {
        push = 20 * (std::max(3 - wf, wf - 4) + std::max(3 - wr, wr - 4));  // 120 = a corner
    }
    const Score material = q * QueenValue + r * RookValue + nb * BishopValue + n * KnightValue + p * PawnValue;
    // Edge and king-distance terms of similar size: both are needed to mate.
    const Score sc = KnownWin + material + push + 20 * (7 - dist);
    white_score = strong == chess::Color::WHITE ? sc : -sc;
    return true;
}

// True when known_ending() answers this position (the search then leaves
// the score uncorrected: correction history would blur the mating terms).
inline bool is_known_ending(const chess::Board& b) {
    chess::Color strong;
    return g_kxk && classify_ending(b, strong) != KnownEnding::None;
}

// Side to move's view before fifty-move scaling; evaluate() ==
// fifty_move_scale(evaluate_unscaled(), halfmove clock).
inline Score evaluate_unscaled(const EvalBoard& board) {
    if (g_kxk) {
        Score w;
        if (known_ending(board, w)) return board.sideToMove() == chess::Color::WHITE ? w : -w;
    }
#if EV_NNUE
    const int bucket = nnue::OutBuckets > 1 ? nnue::output_bucket(board.occ().count()) : 0;
    const int stm = board.sideToMove() == chess::Color::WHITE ? 0 : 1;
    if (board.lazy()) return nnue::evaluate_ptr(board.lazy_acc(stm), board.lazy_acc(stm ^ 1), *nnue::network(), bucket);
    return nnue::evaluate(board.accumulator(), stm, *nnue::network(), bucket);
#else
    const Score s = white_view_unscaled(board, board.terms());
    return board.sideToMove() == chess::Color::WHITE ? s : -s;
#endif
}

// Static evaluation for negamax / PVS: positive = good for the side to move.
inline Score evaluate(const EvalBoard& board) {
    return fifty_move_scale(evaluate_unscaled(board), static_cast<int>(board.halfMoveClock()));
}

// Material + piece-square part of the eval only (the incremental terms, no
// attack / pawn / king-safety terms), side to move's view, with tempo. Cheap:
// used for lazy evaluation in the quiescence search (SP_LAZY).
inline Score fast_eval(const EvalBoard& board) {
    const Terms& t = board.terms();
    const SideTerms& w = t.side[0];
    const SideTerms& b = t.side[1];
    Score s = taper(w.material_mg + w.pst_mg + w.mobility_mg - b.material_mg - b.pst_mg - b.mobility_mg,
                    w.material_eg + w.pst_eg + w.mobility_eg - b.material_eg - b.pst_eg - b.mobility_eg, t.phase());
    if (board.sideToMove() != chess::Color::WHITE) s = -s;
    return EvalUseTempo ? s + Tempo : s;
}

// Same, for a plain chess::Board (terms computed from scratch). Always the
// hand-written eval (tools: tuner, evalcheck, ...).
inline Score evaluate(const chess::Board& board) {
    const Score s = white_view(board, compute(board));
    return board.sideToMove() == chess::Color::WHITE ? s : -s;
}

}  // namespace eval

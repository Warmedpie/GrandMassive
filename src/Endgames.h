// Endgames.h : positions whose result is known from the material alone.
// Used by the eval (eval::known_ending, EV_KXK), datagen (adjudication) and
// the NNUE trainer (drops these positions: the net is never asked about them).
// Needs only Chess.h.

#pragma once

#include <bit>
#include <cstdint>

#include "Chess.h"

namespace eval {

enum class KnownEnding { None, Win, Draw };

inline constexpr std::uint64_t LightSquares = 0x55AA55AA55AA55AAULL;  // a1 is dark
inline bool bishops_on_both_colours(std::uint64_t bishops) {
    return (bishops & LightSquares) && (bishops & ~LightSquares);
}

// Win: one side has a bare king, the other a queen, a rook, bishops on both
// colours, or bishop and knight (pawns allowed); `strong` = the winning side.
// Draw: no pawns, one side a bare king, the other at most one minor piece or
// two knights (KK, KNK, KBK, KNNK).
inline KnownEnding classify_ending(const chess::Board& b, chess::Color& strong) {
    using chess::Color;
    using chess::PieceType;
    if (std::popcount(b.occ().getBits()) > 12) return KnownEnding::None;  // cheap exit
    for (int s = 0; s < 2; ++s) {
        const Color st = s == 0 ? Color::WHITE : Color::BLACK, weak = ~st;
        if (b.us(weak).count() != 1) continue;  // the weak side must be a bare king
        const int q = b.pieces(PieceType::QUEEN, st).count(), r = b.pieces(PieceType::ROOK, st).count();
        const std::uint64_t bish = b.pieces(PieceType::BISHOP, st).getBits();
        const int nb = std::popcount(bish), n = b.pieces(PieceType::KNIGHT, st).count();
        const int p = b.pieces(PieceType::PAWN, st).count();
        strong = st;
        if (q > 0 || r > 0 || bishops_on_both_colours(bish) || (nb > 0 && n > 0)) return KnownEnding::Win;
        if (p == 0 && (nb + n <= 1 || (nb == 0 && n == 2))) return KnownEnding::Draw;
    }
    return KnownEnding::None;
}

}  // namespace eval

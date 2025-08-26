class BitSet:
    __slots__ = ("_mask",)

    def __init__(self, mask: int = 0):
        self._mask = int(mask)

    # ---- 基本操作 ----
    def add(self, x: int) -> "BitSet":
        """要素 x を追加した新しい BitSet を返す"""
        return BitSet(self._mask | (1 << x))

    def remove(self, x: int) -> "BitSet":
        """要素 x を除いた新しい BitSet を返す"""
        return BitSet(self._mask & ~(1 << x))

    def discard(self, x: int) -> "BitSet":
        """要素 x があれば除いた新しい BitSet を返す（なければそのまま）"""
        return self.remove(x) if self.contains(x) else self

    def contains(self, x: int) -> bool:
        return (self._mask >> x) & 1

    def __contains__(self, x: int) -> bool:
        return self.contains(x)

    # ---- 集合演算 ----
    def union(self, other: "BitSet") -> "BitSet":
        return BitSet(self._mask | other._mask)

    def intersection(self, other: "BitSet") -> "BitSet":
        return BitSet(self._mask & other._mask)

    def difference(self, other: "BitSet") -> "BitSet":
        return BitSet(self._mask & ~other._mask)

    def issubset(self, other: "BitSet") -> bool:
        return (self._mask & ~other._mask) == 0

    def issuperset(self, other: "BitSet") -> bool:
        return (other._mask & ~self._mask) == 0

    # ---- その他ユーティリティ ----
    def __len__(self) -> int:
        return self._mask.bit_count()

    def __iter__(self):
        m = self._mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            yield i
            m ^= lsb

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BitSet) and self._mask == other._mask

    def __hash__(self) -> int:
        return hash(self._mask)

    def __repr__(self) -> str:
        elems = ", ".join(str(x) for x in self)
        return f"BitSet({{{elems}}})"

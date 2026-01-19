#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict, Set

SEP = "-" * 60
BIG = "=" * 60

# По условию 3<=|xi|<=5
SIZE_MIN = 3
SIZE_MAX = 5


def parse_row(line: str, n: int) -> List[int]:
    toks = line.strip().split()
    if len(toks) != n:
        raise ValueError(f"Нужно {n} элементов в строке, получено {len(toks)}")
    row: List[int] = []
    for t in toks:
        if t == "*":
            row.append(0)
        else:
            v = int(t)
            if v < 0:
                raise ValueError("Кратности рёбер должны быть >= 0")
            row.append(v)
    return row


def read_matrix_from_file(filename: str) -> List[List[int]]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{filename}' не найден (положи рядом со скриптом).")

    lines = [ln.strip() for ln in raw_lines if ln.strip() != ""]
    if not lines:
        raise ValueError(f"Файл '{filename}' пустой.")

    n = len(lines)
    mat: List[List[int]] = []

    for i, ln in enumerate(lines, start=1):
        toks = ln.split()
        if len(toks) != n:
            raise ValueError(
                f"Матрица должна быть квадратной: строк = {n}, "
                f"а в строке {i} элементов = {len(toks)}.\n"
                f"Проблемная строка {i}: {ln}"
            )
        mat.append(parse_row(ln, n))

    return mat


def read_input() -> Tuple[List[List[int]], int]:
    n_or_file = input("n = ").strip()

    if n_or_file == "matrix.txt":
        print("\nЗагружаю матрицу из matrix.txt ...\n")
        mat = read_matrix_from_file("matrix.txt")
        n = len(mat)
        print(f"ОК: матрица {n}x{n} загружена из matrix.txt\n")
    else:
        n = int(n_or_file)
        print("\nВводи матрицу построчно (числа или '*'), через пробелы.\n")
        mat: List[List[int]] = []
        for i in range(n):
            while True:
                line = input(f"строка {i+1}: ")
                try:
                    row = parse_row(line, n)
                except Exception as e:
                    print(f"Ошибка: {e}. Повтори строку.")
                    continue
                mat.append(row)
                break

    m = int(input("m (лимит суммы внешних связей) = ").strip())
    return mat, m


def w(mat: List[List[int]], a: int, b: int) -> int:
    """Кратность ребра между a и b. Если заполнен только один треугольник — берём максимум."""
    if a == b:
        return 0
    va = mat[a][b]
    vb = mat[b][a]
    return va if va >= vb else vb


def degrees(mat: List[List[int]], verts: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    deg: Dict[int, int] = {}
    nbr: Dict[int, int] = {}
    for v in verts:
        s = 0
        c = 0
        for u in verts:
            if u == v:
                continue
            ww = w(mat, v, u)
            if ww:
                s += ww
                c += 1
        deg[v] = s
        nbr[v] = c
    return deg, nbr


def neighbors_of_piece(mat: List[List[int]], piece: List[int], remaining: Set[int]) -> List[int]:
    cand: Set[int] = set()
    piece_set = set(piece)
    for u in piece:
        for v in remaining:
            if v in piece_set:
                continue
            if w(mat, u, v) > 0:
                cand.add(v)
    return sorted(cand)


def fmt_x(v: int) -> str:
    return f"x{v+1}"


def print_piece_line(step: int, part: int, piece: List[int]) -> None:
    if len(piece) == 1:
        print(f"x{step} {part} = {fmt_x(piece[0])}")
    else:
        inside = ", ".join(fmt_x(v) for v in piece)
        print(f"x{step} {part} = ({inside})")


def print_gamma_line(step: int, part: int, gamma: List[int]) -> None:
    if not gamma:
        print(f"Гx{step} {part} = ()")
        return
    inside = ", ".join(fmt_x(v) for v in gamma)
    print(f"Гx{step} {part} = ({inside})")


def compute_d_for_candidates(
    mat: List[List[int]],
    piece: List[int],
    gamma: List[int],
    deg: Dict[int, int],
) -> List[Tuple[int, int, List[int]]]:
    out: List[Tuple[int, int, List[int]]] = []
    for v in gamma:
        ps = [w(mat, u, v) for u in piece]
        dv = deg[v] - 2 * sum(ps)
        out.append((v, dv, ps))
    return out


def print_table(cand_info: List[Tuple[int, int, List[int]]], deg: Dict[int, int]) -> None:
    print("\nТаблица приращений:")
    print("  xj  | " + " | ".join(f"{v+1}" for v, _, _ in cand_info))

    cells: List[str] = []
    for v, dv, ps in cand_info:
        i = deg[v]
        if len(ps) == 1:
            cells.append(f"{i}-2*{ps[0]}={dv}")
        else:
            cells.append(f"{i}-2*({'+'.join(map(str, ps))})={dv}")
    print("d(xj) | " + " | ".join(cells))
    print()


def print_degrees_block(mat: List[List[int]]) -> None:
    verts = list(range(len(mat)))
    deg, nbr = degrees(mat, verts)

    print("\n" + BIG)
    print("Степени вершин (сумма связей по строкам):")
    print(SEP)
    for v in verts:
        print(f"{fmt_x(v)}: ρ={deg[v]}, соседей={nbr[v]}")
    print(SEP)
    arr = ", ".join(str(deg[v]) for v in verts)
    print(f"ρ(x) = [{arr}]")
    print(BIG + "\n")


def sequential_partition(mat: List[List[int]], m: int) -> List[List[int]]:
    remaining: Set[int] = set(range(len(mat)))
    parts: List[List[int]] = []
    part_id = 1

    print_degrees_block(mat)

    while remaining:
        verts = sorted(remaining)
        deg, nbr = degrees(mat, verts)

        start = min(verts, key=lambda v: (deg[v], nbr[v], v))
        piece: List[int] = [start]
        S = deg[start]

        print(BIG)
        print(f"Кусок {part_id}")
        print(SEP)
        step = 1
        print_piece_line(step=step, part=part_id, piece=piece)

        gamma = neighbors_of_piece(mat, piece, remaining)
        print_gamma_line(step=step, part=part_id, gamma=gamma)
        print(f"Σδ(xg) = {S} (m={m})")
        print(SEP)

        # Если старт уже > m — этот кусок будет одиночным
        if S > m:
            print(f"\nОстанов: Σδ(xg)={S} > m={m} уже на старте.\n")
            parts.append(piece)
            remaining.remove(start)
            print(f"Кусок {part_id} готов: ({fmt_x(start)})")
            print(BIG + "\n")
            part_id += 1
            continue

        while True:
            # размер куска
            if len(piece) >= SIZE_MAX:
                print(f"\nОстанов: |x|={len(piece)} (достигли максимум {SIZE_MAX})")
                break

            gamma = neighbors_of_piece(mat, piece, remaining)
            gamma = [v for v in gamma if v not in piece]
            if not gamma:
                print("\nКандидатов больше нет -> конец куска.")
                break

            cand_info = compute_d_for_candidates(mat, piece, gamma, deg)
            print_table(cand_info, deg)

            # 1) обычные (не превышают m)
            feasible_idx = [i for i, (_, dv, _) in enumerate(cand_info) if S + dv <= m]

            if feasible_idx:
                best_idx = min(feasible_idx, key=lambda i: (cand_info[i][1], i))
                v_best, d_best, _ = cand_info[best_idx]

                print(f"min d(xj) = {d_best} -> берём {fmt_x(v_best)}")

                piece.append(v_best)
                S += d_best
                step += 1

                print_piece_line(step=step, part=part_id, piece=piece)
                gamma2 = neighbors_of_piece(mat, piece, remaining)
                gamma2 = [v for v in gamma2 if v not in piece]
                print_gamma_line(step=step, part=part_id, gamma=gamma2)
                print(f"Σδ(xg) = {S} (m={m})")
                print(SEP)
                continue

            # 2) нет хода, который укладывается в m -> пробуем 1 тестовый прыжок
            if len(piece) > SIZE_MAX - 2:
                print(f"Останов: нет места для тестового шага и восстановления (|x|={len(piece)}, max={SIZE_MAX})")
                break

            # берём "наименее больной" перелёт: минимизируем (S+dv), затем dv, затем порядок
            probe_idx = min(range(len(cand_info)), key=lambda i: (S + cand_info[i][1], cand_info[i][1], i))
            v_probe, d_probe, _ = cand_info[probe_idx]
            S_before = S
            piece_before_len = len(piece)

            print(f"Нет кандидатов с Σδ(xg)+d <= m. ТЕСТОВЫЙ ШАГ: берём {fmt_x(v_probe)} "
                  f"(Σδ станет {S + d_probe} > m)")

            piece.append(v_probe)
            S += d_probe
            step += 1

            print_piece_line(step=step, part=part_id, piece=piece)
            gamma_p = neighbors_of_piece(mat, piece, remaining)
            gamma_p = [v for v in gamma_p if v not in piece]
            print_gamma_line(step=step, part=part_id, gamma=gamma_p)
            print(f"Σδ(xg) = {S} (m={m})")
            print(SEP)

            if not gamma_p:
                # откат
                print("Тестовый шаг не дал кандидатов для восстановления -> ОТКАТ и конец куска.")
                piece.pop()
                S = S_before
                step -= 1
                break

            cand_info_p = compute_d_for_candidates(mat, piece, gamma_p, deg)
            print_table(cand_info_p, deg)

            # восстановление: нужен отрицательный d и чтобы за 1 шаг вернуться в m
            recover_idx = [
                i for i, (_, dv, _) in enumerate(cand_info_p)
                if dv < 0 and S + dv <= m
            ]

            if not recover_idx:
                print("Восстановление не вышло (нет d<0, которое вернёт Σδ<=m за 1 шаг) -> ОТКАТ и конец куска.")
                piece.pop()
                S = S_before
                step -= 1
                if piece_before_len < SIZE_MIN:
                    print(f"Предупреждение: |x|={len(piece)} < {SIZE_MIN}, но по m больше не собрать.")
                break

            best_rec = min(recover_idx, key=lambda i: (cand_info_p[i][1], i))
            v_rec, d_rec, _ = cand_info_p[best_rec]

            print(f"ВОССТАНОВЛЕНИЕ: min d(xj) = {d_rec} -> берём {fmt_x(v_rec)} (вернёмся в m)")

            piece.append(v_rec)
            S += d_rec
            step += 1

            print_piece_line(step=step, part=part_id, piece=piece)
            gamma3 = neighbors_of_piece(mat, piece, remaining)
            gamma3 = [v for v in gamma3 if v not in piece]
            print_gamma_line(step=step, part=part_id, gamma=gamma3)
            print(f"Σδ(xg) = {S} (m={m})")
            print(SEP)
            # дальше продолжаем обычным образом

        parts.append(piece)
        for v in piece:
            remaining.remove(v)

        print(f"\nКусок {part_id} готов: ({', '.join(fmt_x(v) for v in piece)})")
        print(BIG + "\n")
        part_id += 1

    print(BIG)
    print("Итоговая компоновка:")
    for i, p in enumerate(parts, 1):
        print(f"{i}: ({', '.join(fmt_x(v) for v in p)})")
    print(BIG)
    return parts


def main() -> None:
    mat, m = read_input()
    sequential_partition(mat, m)


if __name__ == "__main__":
    main()

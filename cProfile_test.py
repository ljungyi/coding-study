# profile_demo.py


def count_matches_slow(values, queries):

    count = 0

    for q in queries:

        if q in values:
            count += 1

    return count


def build_squares(n):

    result = []

    for i in range(n):
        result.append(i * i)

    return result


def sum_numbers(n):

    total = 0

    for i in range(n):
        total += i

    return total


def main():

    values = list(range(100_000))
    queries = list(range(50_000, 150_000))

    print(
        count_matches_slow(values, queries)
    )

    build_squares(2_000_000)
    sum_numbers(5_000_000)


if __name__ == "__main__":
    main()
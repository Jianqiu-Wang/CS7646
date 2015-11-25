__name__ = '__main__'

import time

def main():
    print nth_prime(1000)

def nth_prime(N):
    current = 1
    prime_count = 0
    while prime_count < N+1:
        prime = True
        for num in range(2,current+1):
            if current % num == 0 and current != num:
                prime = False
        if prime:
            prime_count += 1
            if prime_count == N+1:
                break
        current += 1
    return current

if __name__ == "__main__":
    main()
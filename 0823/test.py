def create_array(n, m):
    # n * m 크기의 2차원 배열 생성
    array = [[0] * m for _ in range(n)]
    
    # 배열을 채울 값 초기화
    current_value = n * m

    # 배열 채우기
    for i in range(m):
        for j in range(n):
            array[j][m - i - 1] = current_value
            current_value -= 1

    # 배열 출력
    for row in array:
        print(' '.join(map(str, row)))

# 입력 받기
n, m = map(int, input().split())

# 배열 생성 및 출력
create_array(n, m)

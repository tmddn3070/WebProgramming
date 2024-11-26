#include <stdio.h>

int main() {
    int N;
    int A, B;

    scanf("%d", &N);

    for (int i = 0; i < N; i++) {
        scanf("%d %d", &A, &B);

        if (A == 1 && B == 1) {
            printf("AND ");
        }

        if (A == 1 || B == 1) {
            printf("OR");
        }

        if (!(A == 1 && B == 1) && !(A == 1 || B == 1)) {
            printf("NONE");
        }

        printf("\n");
    }

    return 0;
}

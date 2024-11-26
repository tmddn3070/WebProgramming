#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double real;
    double imag;
} Complex;

void fft(Complex *x, int n) {
    if (n <= 1) return;

    Complex *even = (Complex *)malloc(n / 2 * sizeof(Complex));
    Complex *odd = (Complex *)malloc(n / 2 * sizeof(Complex));

    for (int i = 0; i < n / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even, n / 2);
    fft(odd, n / 2);

    for (int k = 0; k < n / 2; k++) {
        double t = -2 * M_PI * k / n;
        Complex e = {cos(t), sin(t)};
        Complex temp = {e.real * odd[k].real - e.imag * odd[k].imag, e.real * odd[k].imag + e.imag * odd[k].real};

        x[k].real = even[k].real + temp.real;
        x[k].imag = even[k].imag + temp.imag;
        x[k + n / 2].real = even[k].real - temp.real;
        x[k + n / 2].imag = even[k].imag - temp.imag;
    }

    free(even);
    free(odd);
}

int main() {
    int n;
    scanf("%d", &n);

    Complex *x = (Complex *)malloc(n * sizeof(Complex));
    for (int i = 0; i < n; i++) {
        scanf("%lf", &x[i].real);
        x[i].imag = 0.0;
    }

    fft(x, n);

    for (int i = 0; i < n; i++) {
        printf("%.2f %.2f\n", x[i].real, x[i].imag);
    }

    free(x);
    return 0;
}
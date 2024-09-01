use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    let mut input = stdin.lock().lines();

    let n: usize = input.next().unwrap().unwrap().parse().unwrap();
    let heights: Vec<usize> = input.next().unwrap().unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();

    let total_height: usize = heights.iter().sum();

    if total_height % 3 != 0 {
        println!("NO");
        return;
    }

    let max_twos = total_height / 3;
    let ones_needed = heights.iter().map(|&height| height % 2).sum::<usize>();

    if ones_needed > max_twos {
        println!("NO");
    } else {
        println!("YES");
    }
}

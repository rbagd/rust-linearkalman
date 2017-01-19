# Kalman filtering and smoothing library written in Rust

[![Build Status](https://travis-ci.org/rbagd/rust-linearkalman.svg?branch=master)](https://travis-ci.org/rbagd/rust-linearkalman)

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)

Access documentation for the library [here](http://rbagd.eu/rust-linearkalman/linearkalman/index.html).

Currently, library provides only time-invariant linear Kalman filtering and smoothing technique is known as fixed-interval smoothing (Rauch-Tung-Striebel smoother) which relies on Kalman filter estimates for the entire dataset.

This library relies on [rulinalg] library to implement linear algebra structures and operations and so input data is expected to be a `std::vec::Vec` of [Vector<f64>] objects, i.e. a vector of vectors.

## Example

Below example assumes 3-dimensional measurement data with an underlying 2-dimensional state space model. With the help of a few macros from [rulinalg], a simple attempt to use the library to do Kalman filtering would be as follows.


```rust
#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use rulinalg::vector::Vector;
use linearkalman::KalmanFilter;

fn main() {

  let kalman_filter = KalmanFilter {
    // Process noise covariance
    q: matrix![1.0, 0.1;
               0.1, 1.0],
    // Measurement noise matrix
    r: matrix![1.0, 0.2, 0.1;
               0.2, 0.8, 0.5;
               0.1, 0.5, 1.2],
    // Observation matrix
    h: matrix![1.0, 0.7;
               0.5, 0.7;
               0.8, 0.1],
    // State transition matrix
    f: matrix![0.6, 0.2;
               0.1, 0.3],
    // State variable initial value
    x0: vector![1.0, 1.0],
    // State variable initial covariance
    p0: matrix![1.0, 0.0;
                0.0, 1.0],
  };

  let data: Vec<Vector<f64>> = vec![vector![1.04, 2.20, 3.12],
                                    vector![1.11, 2.33, 3.34],
                                    vector![1.23, 2.21, 3.45]];

  let run_filter = kalman_filter.filter(&data);
  let run_smooth = kalman_filter.smooth(&run_filter.0, &run_filter.1);

  // Print filtered and smoothened state variable coordinates
  println!("filtered.1,filtered.2,smoothed.1,smoothed.2");
  for k in 0..3 {
      println!("{:.6},{:.6},{:.6},{:.6}",
               &run_filter.0[k].x[0], &run_filter.0[k].x[1],
               &run_smooth[k].x[0], &run_smooth[k].x[1])
  }
}
```

`examples` directory contains code sample which allows to import data from a CSV file and returns filtered and smoothed data to `stdout`.

## License

This project is licensed under GPL3.

[rulinalg]: https://github.com/AtheMathmo/rulinalg
[Vector<f64>]: https://athemathmo.github.io/rulinalg/doc/rulinalg/vector/struct.Vector.html

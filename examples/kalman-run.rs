#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use rulinalg::vector::Vector;
use linearkalman::KalmanFilter;

fn main() {

  let kalman_filter = KalmanFilter {
    // State covariance matrix
    q: matrix![1.0, 0.1;
               0.1, 1.0],
    // Process covariance matrix
    r: matrix![1.0, 0.2, 0.1;
               0.2, 0.8, 0.5;
               0.1, 0.5, 1.2],
    // State-dependence matrix
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

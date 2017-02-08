#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use rulinalg::vector::Vector;
use linearkalman::KalmanFilter;

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-7,
                                  "{} is not approximately equal to {}", *a, *b);
        })
}

#[test]
fn smoothing_values() {

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

    let res_f = kalman_filter.filter(&data);
    let res_s = kalman_filter.smooth(&res_f.0, &res_f.1);

    // Based on reference estimates from pykalman module in Python
    //
    // Test smoothing estimates
    assert_approx_eq!(res_s[0].x[0], 1.51225349);
    assert_approx_eq!(res_s[1].x[0], 1.69965529);
    assert_approx_eq!(res_s[2].x[0], 1.68756582);

    assert_approx_eq!(res_s[0].x[1], 0.77961369);
    assert_approx_eq!(res_s[1].x[1], 0.46657133);
    assert_approx_eq!(res_s[2].x[1], 0.33374625);

    // Smoothed covariance
    // t = 1
    assert_approx_eq!(res_s[0].p.data()[0], 0.43626195);
    assert_approx_eq!(res_s[0].p.data()[1], -0.17509815);
    assert_approx_eq!(res_s[0].p.data()[2], -0.17509815);
    assert_approx_eq!(res_s[0].p.data()[3], 0.54651832);
    // t = 2
    assert_approx_eq!(res_s[1].p.data()[0], 0.44567712);
    assert_approx_eq!(res_s[1].p.data()[1], -0.15752478);
    assert_approx_eq!(res_s[1].p.data()[2], -0.15752478);
    assert_approx_eq!(res_s[1].p.data()[3], 0.53925906);
    // t = 3
    assert_approx_eq!(res_s[2].p.data()[0], 0.47780773);
    assert_approx_eq!(res_s[2].p.data()[1], -0.1498213);
    assert_approx_eq!(res_s[2].p.data()[2], -0.1498213);
    assert_approx_eq!(res_s[2].p.data()[3], 0.54913824);

}

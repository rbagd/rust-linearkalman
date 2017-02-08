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
fn filter_values() {

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

    let res = kalman_filter.filter(&data);

    // Based on reference estimates from FKF package in R and
    // pykalman module in Python
    //
    // Test that filtered estimates are correct
    assert_approx_eq!(res.0[0].x[0], 1.35635067);
    assert_approx_eq!(res.0[1].x[0], 1.55174759);
    assert_approx_eq!(res.0[2].x[0], 1.68756582);

    assert_approx_eq!(res.0[0].x[1], 0.77081876);
    assert_approx_eq!(res.0[1].x[1], 0.46023087);
    assert_approx_eq!(res.0[2].x[1], 0.33374625);
    // Test that filtered covariance estimates are correct
    // t = 1
    assert_approx_eq!(res.0[0].p.data()[0], 0.46616058);
    assert_approx_eq!(res.0[0].p.data()[1], -0.16948804);
    assert_approx_eq!(res.0[0].p.data()[2], -0.16948804);
    assert_approx_eq!(res.0[0].p.data()[3], 0.55648347);
    // t = 2
    assert_approx_eq!(res.0[1].p.data()[0], 0.47692309);
    assert_approx_eq!(res.0[1].p.data()[1], -0.15019802);
    assert_approx_eq!(res.0[1].p.data()[2], -0.15019802);
    assert_approx_eq!(res.0[1].p.data()[3], 0.54950433);
    // t = 3
    assert_approx_eq!(res.0[2].p.data()[0], 0.47780773);
    assert_approx_eq!(res.0[2].p.data()[1], -0.1498213);
    assert_approx_eq!(res.0[2].p.data()[2], -0.1498213);
    assert_approx_eq!(res.0[2].p.data()[3], 0.54913824);

    // Test that predicted estimates are correct
    assert_approx_eq!(res.1[0].x[0], 1.0);
    assert_approx_eq!(res.1[1].x[0], 0.9679742);
    assert_approx_eq!(res.1[2].x[0], 1.0230947);
    assert_approx_eq!(res.1[3].x[0], 1.0792887);

    assert_approx_eq!(res.1[0].x[1], 1.0);
    assert_approx_eq!(res.1[1].x[1], 0.3668807);
    assert_approx_eq!(res.1[2].x[1], 0.2932440);
    assert_approx_eq!(res.1[3].x[1], 0.2688805);


}

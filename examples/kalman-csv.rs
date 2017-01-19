extern crate csv;
extern crate linearkalman;
extern crate rulinalg;

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use linearkalman::KalmanFilter;

fn main() {

    let args: Vec<String> = std::env::args().collect();
    let csv_file = &args[1];

    if !std::path::Path::new(csv_file).exists() {
        panic!("{} file does not exist.", csv_file)
    }

    let kalman_filter = KalmanFilter {
        q: Matrix::from_diag(&vec![1.0, 1.0]),    // State covariance
        r: Matrix::from_diag(&vec![1.0, 1.0]),    // Measurement covariance
        h: Matrix::from_diag(&vec![1.0, 1.0]),    // State-dependence matrix
        f: Matrix::from_diag(&vec![1.0, 1.0]),    // State transition matrix
        x0: Vector::new(vec![1.0, 1.0]),          // State variable initial value
        p0: Matrix::from_diag(&vec![1.0, 1.0]),   // State covariance initial value
    };

    // Container to store data
    let mut data: Vec<Vector<f64>> = Vec::new();

    // Loop over the CSV file and gather all data
    let mut reader = csv::Reader::from_file(csv_file).unwrap();
    for record in reader.decode() {
        let (x1, x2): (f64, f64) = record.unwrap();
        data.push(Vector::new(vec![x1, x2]));
    }

    let filtered = kalman_filter.filter(&data);
    let smoothed = kalman_filter.smooth(&filtered.0, &filtered.1);

    println!("filtered.x1,filtered.x2,smoothed.x1,smoothed.x2");
    for k in 0..filtered.0.len() {
        println!("{},{},{},{}",
                 &filtered.0[k].x[0], &filtered.0[k].x[1],
                 &smoothed[k].x[0], &smoothed[k].x[1]
        );
    }
}

use candle_core::Error;
use candle_core::{Device, Tensor};

fn main() -> Result<(), Error> {
    let a = Tensor::randn(0f32, 1., (2, 3), &Device::Cpu)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &Device::Cpu)?;

    let c = a.matmul(&b)?;
    println!("{c}");

    Ok(())
}

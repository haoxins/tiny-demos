use candle_core::Error;
use candle_core::{Device, Tensor};

fn main() -> Result<(), Error> {
    let a = Tensor::randn(0f32, 1., (2, 3), &Device::Cpu)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &Device::Cpu)?;

    let c = a.matmul(&b)?;
    println!("{c}");

    let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;

    let c = a.matmul(&b)?;
    println!("{c}");

    Ok(())
}

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let weight = Tensor::zeros((100, 784), DType::F32, &device)?;
    let bias = Tensor::zeros((100,), DType::F32, &device)?;
    let first = Linear::new(weight, Some(bias));
    let weight = Tensor::zeros((10, 100), DType::F32, &device)?;
    let bias = Tensor::zeros((10,), DType::F32, &device)?;
    let second = Linear::new(weight, Some(bias));
    let model = Model { first, second };

    let dummy_image = Tensor::zeros((1, 784), DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}

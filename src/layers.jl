gpu(layer::Binary) = Binary(gpu(layer.par))
cpu(layer::Binary) = Binary(cpu(layer.par))

gpu(layer::Spin) = Spin(gpu(layer.par))
cpu(layer::Spin) = Spin(cpu(layer.par))

gpu(layer::Potts) = Potts(gpu(layer.par))
cpu(layer::Potts) = Potts(cpu(layer.par))

gpu(layer::Gaussian) = Gaussian(gpu(layer.par))
cpu(layer::Gaussian) = Gaussian(cpu(layer.par))

gpu(layer::ReLU) = ReLU(gpu(layer.par))
cpu(layer::ReLU) = ReLU(cpu(layer.par))

gpu(layer::dReLU) = dReLU(gpu(layer.par))
cpu(layer::dReLU) = dReLU(cpu(layer.par))

gpu(layer::pReLU) = pReLU(gpu(layer.par))
cpu(layer::pReLU) = pReLU(cpu(layer.par))

gpu(layer::xReLU) = xReLU(gpu(layer.par))
cpu(layer::xReLU) = xReLU(cpu(layer.par))

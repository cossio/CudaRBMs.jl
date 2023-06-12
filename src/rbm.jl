gpu(rbm::RBM) = RBM(gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w))
cpu(rbm::RBM) = RBM(cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w))

gpu(∂::∂RBM) = ∂RBM(gpu(∂.visible), gpu(∂.hidden), gpu(∂.w))
cpu(∂::∂RBM) = ∂RBM(cpu(∂.visible), cpu(∂.hidden), cpu(∂.w))

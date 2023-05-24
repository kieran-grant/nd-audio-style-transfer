from src.utils.utils import *
from src.wrappers.dafx_wrapper import DAFXWrapper


# ==== SPSA with DAFX ====
class SPSABatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, dafx: DAFXWrapper, epsilon: float):
        ctx.save_for_backward(x, y)
        ctx.dafx = dafx
        ctx.epsilon = epsilon

        device = x.device

        x, y = x.cpu(), y.cpu()

        z = []

        # check if batch size == 1
        if len(x.shape) == 1:
            z.append(dafx.apply(x, y))
        else:
            for i in range(x.size()[0]):
                z.append(dafx.apply(x[i], y[i]))

        return torch.stack(z).to(device)

    @staticmethod
    def backward(ctx, upstream_grad):
        # Get inputs from context and move to CPU
        x, y = ctx.saved_tensors
        device = x.device

        x, y, upstream_grad = x.cpu(), y.cpu(), upstream_grad.cpu()

        # Get hyperparameters from context
        dafx = ctx.dafx
        epsilon = ctx.epsilon

        def _grad(dye, xe, ye):
            # Grad w.r.t x
            delta_Kx = rademacher(xe.shape)
            J_plus = dafx.apply(xe + (epsilon * delta_Kx), ye)
            J_minus = dafx.apply(xe - (epsilon * delta_Kx), ye)
            d_dx = (J_plus - J_minus) / (2.0 * epsilon)

            # Grad w.r.t y
            delta_Ky = rademacher(ye.shape)

            params_plus = ye + (epsilon * delta_Ky)
            J_plus = dafx.apply(xe, params_plus)

            params_minus = ye - (epsilon * delta_Ky)
            J_minus = dafx.apply(xe, params_minus)

            grad_param = J_plus - J_minus

            downstream_dy = torch.zeros_like(ye)

            # iterate over parameters
            for i in range(ye.size()[0]):
                d_dy = grad_param / (2.0 * epsilon * delta_Ky[i])
                # add entry to output jacobian
                downstream_dy[i] = torch.dot(dye, d_dy)

            downstream_dx = d_dx * dye

            return downstream_dx, downstream_dy

        dx = []
        dy = []

        for i in range(upstream_grad.size()[0]):
            vecJxe, vecJye = _grad(upstream_grad[i], x[i], y[i])
            dx.append(vecJxe)
            dy.append(vecJye)

        return torch.stack(dx).to(device), torch.stack(dy).to(device), None, None, None

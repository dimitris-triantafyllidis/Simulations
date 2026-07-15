The Kadomtsev–Petviashvili equation reads:

$$
(u_t+6uu_x+u_{xxx})_x+3\sigma^2u_{yy}=0, \quad \sigma^2=\pm 1,
$$

or, for $u$ approaching zero sufficiently quickly at infinity,

$$
u_t=-3\frac{\partial{(u^2)}}{\partial{x}}-u_{xxx}-3\sigma^2\int_{-\infty}^x u_{yy}\, dx,
$$

The Fourier transform with respect to $x$ is:

$$
    \frac{\partial{\mathcal{F}_{x}(u)}}{\partial{t}}=-3ik_x\mathcal{F}_{x}\left(u^2\right)+ik_x^3\mathcal{F}_{x}(u) - 3\sigma^2\left(\frac{\mathcal{F}_x(u_{yy})}{ik_x}+\pi\mathcal{F}_x(u_{yy})(0)\delta(k_x)\right),
$$

where we have to require that $\mathcal{F}_x(u_{yy})(0, y) = 0$:

$$
    \frac{\partial{\mathcal{F}_{x}(u)}}{\partial{t}}=-3ik_x\mathcal{F}_{x}\left(u^2\right)+ik_x^3\mathcal{F}_{x}(u) - 3\sigma^2\frac{\mathcal{F}_x(u_{yy})}{ik_x},
$$

and taking the Fourier transform with respect to $y$, we have:

$$
    \frac{\partial{\mathcal{F}_{xy}(u)}}{\partial{t}}=-3ik_x\mathcal{F}_{xy}\left(u^2\right)+\left(ik_x^3 - 3\sigma^2\frac{ik_{y}^2}{k_x}\right)\mathcal{F}_{xy}(u),
$$

and rewriting the middle term in terms of $u$ to facilitate a time-stepping numerical solution,

$$
    \frac{\partial{\mathcal{F}_{xy}(u)}}{\partial{t}}=-3ik_x\mathcal{F}_{xy}\left(\left(\mathcal{F}^{-1}_{xy}\left(\mathcal{F}_{xy}\left(u\right)\right)\right)^2\right)+\left(ik_x^3-3\sigma^2\frac{ik_y^2}{k_x}\right)\mathcal{F}_{xy}(u).
$$

Setting $\hat{u}\equiv\mathcal{F}_{xy}(u)$, the evolution operator for the (exactly solvable) dispersive subproblem is defined by:

$$
\Phi_{L}(\hat{u}, \Delta t)=\hat{u}e^{\left(ik_x^3-3\sigma^2\frac{ik_y^2}{k_x}\right)\Delta t}
$$

For the nonlinear subproblem:

$$
\Phi_{N}\left(\hat{u}, \Delta t\right) \approx RK4\left(-3ik_x\mathcal{F}_{xy}\left(\mathcal{F}^{-1}_{xy}\left(\hat{u}\right)^2\right), \hat{u}, \Delta t\right)
$$

The numerical scheme for one full step with Strang operator splitting is:

$$
\hat{u}^{(1/3)} = \Phi_{L}\left(\hat{u}^{(0)}, \frac{\Delta t}{2}\right)
$$

$$
\hat{u}^{(2/3)} = \Phi_{N}(\hat{u}^{(1/3)}, \Delta t)
$$


$$
\hat{u}^{(1)} = \Phi_{L}\left(\hat{u}^{(2/3)}, \frac{\Delta t}{2}\right)
$$


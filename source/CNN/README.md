### Batch Normalization Gradient

$$
z_{il}=f(y_{il})
$$

$$
y_{il}=\alpha_{i}\frac{x_{il}-m_{i}}{\sqrt{\varepsilon+v_{i}}}+\beta_{i}
$$

$$
m_{i}=\sum_{j}x_{ij}/N
$$

$$
v_{i}=\sum_{j}(x_{ij}-m_{i})^{2}/N
$$

$$
\begin{split}
\frac{\partial{L}}{\partial{x_{ij}}}
&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}\frac{\partial{z_{kl}}}{\partial{y_{kl}}}\frac{\partial{y_{kl}}}{\partial{x_{ij}}}\\
&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}f'(y_{kl})\frac{\partial{y_{kl}}}{\partial{x_{ij}}}\delta_{ki}\\
&=\sum_{l}\frac{\partial{L}}{\partial{z_{il}}}f'(y_{il})\frac{\partial{y_{il}}}{\partial{x_{ij}}}\\
&=\sum_{l}\frac{\partial{L}}{\partial{z_{il}}}f'(y_{il})\frac{\alpha_{i}}{\varepsilon+v_{i}}\left[\left(\delta_{lj}-\frac{1}{N}\right)\sqrt{\varepsilon+v_{i}}-\frac{x_{il}-m_{i}}{2\sqrt{\varepsilon+v_{i}}}\frac{\partial{v_{i}}}{\partial{x_{ij}}}\right]\\
\end{split}
$$

$$
\begin{split}
\frac{\partial{v_{i}}}{\partial{x_{ij}}}&=\frac{\partial}{\partial{x_{ij}}}\left[\frac{1}{N}\sum_{k}\left(x_{ik}-m_{i}\right)^{2}\right]\\
&=\frac{1}{N}\sum_{k}2\left(x_{ik}-m_{i}\right)\left(\delta_{kj}-\frac{1}{N}\right)\\
&=\frac{2}{N}\left(x_{ij}-m_{i}\right)-\frac{2}{N^{2}}\sum_{k}\left(x_{ik}-m_{i}\right)\\
&=\frac{2}{N}\left(x_{ij}-m_{i}\right)
\end{split}
$$

$$
\begin{split}
\frac{\partial{L}}{\partial{\beta_{i}}}&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}\frac{\partial{z_{kl}}}{\partial{y_{kl}}}\frac{\partial{y_{kl}}}{\partial{\beta_{i}}}\\&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}f'(y_{kl})\delta_{ki}=\sum_{l}\frac{\partial{L}}{\partial{z_{il}}}f'(y_{il})
\end{split}
$$

$$
\begin{split}
\frac{\partial{L}}{\partial{\alpha_{i}}}&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}\frac{\partial{z_{kl}}}{\partial{y_{kl}}}\frac{\partial{y_{kl}}}{\partial{\alpha_{i}}}\\
&=\sum_{kl}\frac{\partial{L}}{\partial{z_{kl}}}f'(y_{kl})\delta_{ki}\frac{x_{kl}-m_{i}}{\sqrt{\varepsilon+v_{i}}}\\
&=\sum_{l}\frac{\partial{L}}{\partial{z_{il}}}f'(y_{il})\frac{x_{il}-m_{i}}{\sqrt{\varepsilon+v_{i}}}
\end{split}
$$

### Softmax Gradient

$$
y_{i}=\frac{e^{x_{i}+C}}{\sum_{k} e^{x_{k}+C}}
$$

$$
\begin{split}
\frac{\partial{y_{i}}}{\partial{x_{j}}}
&=\frac{\delta_{ij}e^{x_{i}+C}\sum_{k} e^{x_{k}+C}-e^{x_{i}+C}e^{x_{j}+C}}{\left(\sum_{k} e^{x_{k}+C}\right)^{2}}\\
&=\delta_{ij}y_{i}-y_{i}y_{j}\\
&=y_{i}(\delta_{ij}-y_{j})
\end{split}
$$

$$
\begin{split}
\frac{\partial{L}}{\partial{x_{j}}}
&=\sum_{l}\frac{\partial{L}}{\partial{y_{l}}}\frac{\partial{y_{l}}}{\partial{x_{j}}}\\
&=\sum_{l}\frac{\partial{L}}{\partial{y_{l}}}y_{l}(\delta_{lj}-y_{j})\\
&=\frac{\partial{L}}{\partial{y_{j}}}y_{j}-\sum_{l}\frac{\partial{L}}{\partial{y_{l}}}y_{l}y_{j}
\end{split}
$$
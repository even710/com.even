# 分类器和逻辑回归

分类器问题，跟回归问题相似，只是其中的$y$值是一些离散的小的数字。现在，我们先处理一个二进制问题。令$y$只能取两个值，0和1。

现在，要处理一个垃圾邮件过滤的问题，$x^{(i)}$ 是一封邮件的一些特征，如果邮件是垃圾邮件，则 $y$ 等于1；反之，$y$ 等于0。0被称为负类，1被称为正类。有时也用“-”和“+”分别表示。给定 $x^{(i)}$ ，与之匹配的 $y^{(i)}$ 被称为标签。

## 1 逻辑回归

用线性回归来处理分类器问题，效果很差。因此，需要重新定义 $h_{\theta}(x)$ 。

$$
h_{\theta}(x)=g(\theta^T x)=\frac{1}{1+e^{-\theta^T x}}
$$

在这里，

$$
g(z)=\frac{1}{1+e^{-z}}
$$
<!-- ![](http://latex.codecogs.com/gif.latex?g(z)=\frac{1}{1+e^{-z}}) -->
被称为**逻辑函数**或者**sigmoid函数**，坐标图：

![sigmoid](https://raw.githubusercontent.com/even710/com.even/master/images/sigmoid.png)

注意到当$z$ 无穷大时，$g(z)$ 趋向于1，反之，$g(z)$ 趋向于0。因此，$h(x)$ 的值在0到1之间。现在得公式：$\theta^Tx=\theta_0+\begin{matrix}\sum_{j=1}^n \theta_jx_j\end{matrix}$ 其中，$x_0=1$ 。

求$g'$:

$$
\begin{align}g'(z)
&=\frac{d}{dz}\frac{1}{1+e^{-z}}\\
&=\frac{1}{(1+e^{-z})^2}\big(1-\frac{1}{(1+e^{-z})}\big)\\
&=g(z)(1-g(z))
\end{align}
$$

假设：

$$
P(y=1|x;\theta)=h_{\theta}(x)
$$

$$
P(y=0|x;\theta)=1-P(y=1|x;\theta)
$$

把上式总结起来就是：

$$
p(y|x;\theta)=(h_{\theta}(x))^y(1-h_{\theta})^{1-y}
$$

假设 $m$ 个训练集是独立生成的，那么，参数的似然是：

$$\begin{align}L(\theta)
&=p(\overrightarrow y|X;\theta)\\
&=\coprod_{i=1}^m p(y^{(i)}|x^{(i)};\theta)\\
&=\coprod_{i=1}^m (h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}
\end{align}$$

把 $L(\theta)$ 转换成对数似然，以便求最大化：

$$\begin{align}\ell(\theta)
&=log L(\theta)\\
&=\sum_{i=1}^m y^{(i)}log h(x^{(i)}) + (1-y^{(i)})log(1-h(x^{(i)}))
\end{align}$$

令 $\ell)(\theta)$ 对 $\theta$ 求导(结合公式： $g'(z)=g(z)(1-g(z))$ )：

$$\begin{align} \frac{\partial}{\partial \theta_j}
&=\left(y\frac{1}{g(\theta^T x)}-(1-y)\frac{1}{1-g(\theta^T x)}\right)\frac{\partial}{\partial \theta_j}g(\theta^T x)\\
&=\left(y\frac{1}{g(\theta^T x)}-(1-y)\frac{1}{1-g(\theta^T x)}\right)g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^T x\\
&=\left(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx)\right)x_j\\
&=(y-h_{\theta}(x))x_j
\end{align}$$

在处理线性回归的损失函数时，要极大似然值，所以要使损失函数 $\frac{1}{2} \sum_{i=1}^m(y^{(i)}-\theta^Tx^{(i)})^2$ 最小化，因此，线性回归的更新规则是 $\theta:=\theta - \alpha \nabla_{\theta}J(\theta)$ ，而在逻辑回归中，损失函数 $J(\theta)=\ell(\theta)$ ，要极大似然值，则要极大损失函数，所以更新规则变成：$\theta:=\theta + \alpha \nabla_{\theta}\ell(\theta)$ （梯度上升法）:

$$\theta_j := \theta_j + \alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}$$

## 2 感知器学习算法

sigmoid函数是当 $z$ 无穷大时，$g$ 趋向于1，当 $z$ 无穷小时，$g$ 趋向于0，换个角度看，相当于：

$$g(z)=\begin{cases}
1,  & \mbox{if }z \ge 0 \\
0, & \mbox{if }z \leq 0
\end{cases}$$

令 $h_{\theta}(x)=g(\theta^Tx)$ 则更新规则：

$$\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}$$

这就是感知器学习算法。

尽管感知器可能和我们之前讲过的其他算法相似，它实际上是一种和逻辑回归和最小二乘法线性回归非常不同的算法。

概率解释和推导出最大似然估计算法对于感知器来说都比较困难。

## 3 牛顿法最大化 $\ell(\theta)$ 的算法

假设：

$f:\mathbb{R} \mapsto \mathbb{R}$

找出使 $f(\theta)=0$ 的 $\theta$ 值

$\theta \in \mathbb{R}$

牛顿法公式：

$$\theta := \theta - \frac{f(\theta)}{f'(\theta)}$$

![](https://raw.githubusercontent.com/even710/com.even/master/images/newton.png)

左图是 $f(\theta)$ 函数，现在要求 $f(\theta)=0$ 的对应的 $\theta$ 值。先随机取一点为 $\theta_0$ ，然后 $f(\theta_0)$ 对 $\theta_0$ 求导，即作出切线，切线与 $x$ 轴相交点为 $\theta_1$ 然后 $\theta_1$ 对 $\theta_1$ 求导，作切线， 切线与 $x$ 轴交点为 $\theta_2$ ，重复操作，直到 $f(\theta)$ 等于0。

然而，如果用此公式求 $\ell$ 的极大值，在接近极大值时， $\ell'(\theta)$ 会等于0，分母不能为0，因此，令  $f(\theta)=\ell'(\theta)$ ，用同样的算法来极大化 $\ell$

$$\theta:=\theta-\frac{\ell'(\theta)}{\ell''(\theta)}$$

在逻辑回归中， $\theta$ 是一个向量值，用牛顿法设置时需要新的公式（Newton-Raphson）：

$$\theta:=\theta-H^{(-1)}\nabla_{\theta}\ell(\theta)$$

$\nabla_{\theta}\ell(\theta)$ 是 $\ell(\theta)$ 对 $\theta$ 的偏导数， $H$ 是 $(n \times n)$ 矩阵（如果包括截距的话，就是 $(n+1 \times n+1)$ ），也叫海塞矩阵(**Hessian**)，它的元素是,

$$H_{ij}=\frac{\partial^2\ell(\theta)}{\partial\theta_i\partial\theta_j}$$

牛顿法通常比批量梯度下降要更快收敛，而且需要更少的迭代来接近最小值，只是牛顿法的一次迭代特别费时，因此海塞矩阵的 $n$ 太大了。如果用牛顿法应用到逻辑回归中，即要极大化逻辑回归的对数似然函数 $\ell(\theta)$ ，这时结果函数被称为 **Fisher scoring**

# 广义线性模型（Generalized Linear Models）

线性回归：

$y\mid x; \theta \sim \mathcal{N}(\mu, \sigma^2)$

逻辑回归(分类问题)：

$y\mid x; \theta \sim Bernoulli(\phi)$

它们都属于广义线性模型（GLM）

## 4 指数分布族

如果一个分布可以用下面公式表示，则说明它是指数分布族的：

$$p(y;\eta)=b(y)exp(\eta^TT(y)-a(\eta))$$

$\eta$ 表示自然参数，也叫**canonical parameter**

$T(y)$ 表示充分统计量（**sufficient statistic**）。 一般情况下，$T(y)=y$ 。

$a(\eta)$ 表示对数划分函数 (**log partition function**)

$e^{-a(\eta)}$ 表示归一化常量，以确保分布 $p(y;\eta)$ 总和等于1

现在求出Bernoulli和Gaussian分布对应的指数族分布。

均值为 $\phi$ 的Bernoulli分布: $Bernoulli(\phi)$ 指定了$y$的分布： $y \in {0,1}$，所以有：

$$p(y=1;\phi)=\phi$$

$$p(y=0;\phi)=1-\phi$$

**Bernoulli分布：**

$$\begin{align}p(y;\phi)
&=\phi^y(1-\phi)^{1-y}\\
&=exp(ylog\phi+(1-y)log(1-\phi))\\
&=exp\left(\left(log\left(\frac{\phi}{1-\phi}\right)\right)y+log(1-\phi)\right)
\end{align}$$

令 $\eta=log(\phi/(1-\phi))$ ，所以 $\phi=1/(1+e^{-\eta})$ 。该公式跟sigmoid函数很像。

然后，令：

$$T(y)=y$$

$$\begin{align}a(\eta)&=-log(1-\phi)\\&=log(1+e^{\eta})\end{align}$$

$$b(y)=1$$

因此，使用合适的T，a和b，可以使得Bernoulli分布用指数族公式表示。

**高斯(Gaussian)分布：**

在推导线性回归时， $\sigma^2$ 不影响 $\theta$ 和 $h_{\theta}(x)$ 的最终值，因此，可以随意设置 $\sigma^2$ 值，令  $\sigma^2=1$ 方便后面的计算。

$$\begin{align}p(y;\mu)
&=\frac{1}{\sqrt{2\pi}}exp\left(-\frac{1}{2}(y-\mu)^2\right)\\
&=\frac{1}{\sqrt{2\pi}}exp\left(-\frac{1}{2}y^2\right)*exp(\mu y-\frac{1}{2}\mu^2)
\end{align}$$

因此，高斯分布在指数族的表示：

$$\eta=\mu$$

$$T(y)=y$$

$$\begin{align}a(\eta)&=\mu^2/2\\&=\eta^2/2\end{align}$$

$$b(\eta)=(1/\sqrt{2\pi})exp(-y^2/2)$$

除了bernoulli分布和高斯分布外。还有很多分布是属于指数族。如多项式分布，Poisson分布，gamma分布。

## 5 构造GLMs

现在有一个问题，假如你要建立一个模型来评估每天到你店里的人的数量 $y$ ,用Poisson分布可以很好地建立一个模型，而Poisson分布属于指数族，因此，我们可以应用广义线性模型。

现在，来构造一个GLMs。

为了得到一个问题的GLM，我们需要对给定 $x$ 的 $y$ 条件分布和模型有三个假定。

1， $y\mid x;\theta \sim ExponentialFamily(\eta)$ ，给定 $x$ 和 $\theta$ ，$y$ 的分布服从一些指数族分布，以 $\eta$ 为参数。

2， 给定 $x$，我们的目标是预测给定 $x$ 的 $T(y)$ 的期望值( $\mu$ )。一般， $T(y)=y$ ，所以这意味着我们的预测 $h(x)$ 由 $h(x)=E[y\mid x]$ 输出（如逻辑回归中的 $h_{\theta}(x)=p(y=1 \mid x; \theta)=0*p(y=0\mid x; \theta)+1*p(y=1\mid x;\theta)=E[y\mid x;\theta]$ ）。

3,自然参数 $\eta$ 和输入 $x$ 是线性关系的。 $\eta=\theta^Tx$

## 5.1 普通最小二乘法

普通最小二乘法是GLM族的一个特例。

考虑到目标变量 $y$ （在GLM术语中也叫响应变量）是连续的，因此，以高斯分布($\mathbb{N}(\mu,\sigma^2)$ )来建模 $y$ 的连续分布，这里的 $\mu$ 依赖于 $x$ 。

上面提到，高斯分布转成指数分布后， $\mu=\eta$ ，所以，根据构造GLMs的三个假设，得出：

$$\begin{align} h_{\theta}(x)
&=E[y\mid x;\theta]\\
&=\mu\\
&=\eta\\
&=\theta^Tx\\
\end{align}$$

## 5.2 逻辑回归

逻辑回归的 $y$ 服从Bernoulli分布， 在Bernoulli分布转指数族分布时，得出 $\phi=1/(1+e^{-\eta})$ 。此外，如果 $Y \mid x;\theta\sim Bernoulli(\phi)$ ，那么 $E[y\mid x;\theta]=\phi$  因此：

$$\begin{align}h_{\theta}(x)
&=E[y\mid x;\theta]\\
&=\phi\\
&=1/1(1+e^{-\eta})\\
&=1/(1+e^{-\theta^Tx})
\end{align}$$

这就是为什么逻辑函数是 $1/(1+e^{-z})$ 的原因：

**一旦评估以 $x$ 为条件的 $y$ 是Bernoulli分布，它是由GLMs的定义和指数族分布得出的结果。**

另外， $g$ 函数以自然参数的分布（ $g(\eta)=E[T(y);\eta]$ ），被称为 规范响应函数。 $g^{-1}$ 被称为规范连接函数。

**响应函数：** $\eta$ 是概率分布的某个参数（如 $\phi,\mu$ 等）的函数 $g$ ，例如 $\phi=g(\eta)$ ，这时把根据广义线性模型的假设三： $\eta=\theta^Tx$ 代入连接函数，即可得到 $h_{\theta}(x)$ 。

**连接函数：** 用于关联指数分布族中的未知参数 $\eta$ 和要求的权重向量 $\theta$ ，是响应函数的反函数，例如 $\eta = g^{-1}(\phi)$ 。

## 5.3 Softmax回归

分类垃圾邮件时，预测值 $y$ 只能是 (0,1)， 而现在令 $y \in \left\{1,2,\dots,k\right\}$ ，这样的模型是根据多元项分布建立的。

现在推出一个GLM来建模多元项。

首先，把多元项作为指数族分布表示。

要参数多元项的 $k$ 种可能输出，一种方法是用 $k$ 个参数 $\phi_1,\dots,\phi_k$ ，但是这样的话会有点冗余，因为 $\begin{matrix}\sum_{i=1}^k \phi_i=1\end{matrix}$ ，所以最后一项 $\phi_k$ 完全可以用 $\phi_k = 1-\begin{matrix}\sum_{i=1}^{k-1}\phi_i\end{matrix}$ 表示。所以，我们只需要参数化 $\phi_1,\dots,\phi_{k-1}$。

把多项式表示为指数族分布，需要定义 $T(y)\in \mathbb{R}^{k-1}$ ：

$$T(1)=\begin{bmatrix}1\\0\\0\\\vdots\\0\end{bmatrix},T(2)=\begin{bmatrix}0\\1\\0\\\vdots\\0\end{bmatrix},T(3)=\begin{bmatrix}0\\0\\1\\\vdots\\0\end{bmatrix},\dots,T(k-1)=\begin{bmatrix}1\\0\\0\\\vdots\\1\end{bmatrix},T(k)=\begin{bmatrix}0\\0\\0\\\vdots\\0\end{bmatrix}，$$

$T(y)$ 是一个 k-1 维向量。而不再是一个实数，所以公式 $T(y)=y$ 不成立。

**令 $(T(y))_i$ 作为 $T(y)$的第 i 个元素。**

**符号表示**

$$(T(y))_i = 1\left\{y=i\right\}$$

其中 $1\left\{True\right\}=1,1\left\{False\right\}=0$

进一步的：

$$E[(T(y))_i]=P(y=i)=\phi_i$$

多元项的指数族：

$$\begin{align}p(y;\phi)
&=\phi_1^{1\left\{y=1\right\}}\phi_2^{1\left\{y=2\right\}}\dots\phi_k^{1\left\{y=k\right\}}\\
&=\phi_1^{1\left\{y=1\right\}}\phi_2^{1\left\{y=2\right\}}\dots\phi_k^{1-\begin{matrix}\sum_{i=1}^{k-1}1\left\{y=i\right\}\end{matrix}}\\
&=\phi_1^{(T(y))_1}\phi_2^{(T(y))_2}\dots\phi_k^{1-\begin{matrix}\sum_{i=1}^{k-1}(T(y))_i\end{matrix}}\\
&=exp((T(y))_1log(\phi_1)+(T(y))_2log(\phi_2)+\dots+\left(1-\begin{matrix}\sum_{i=1}^{k-1}(T(y))_i\end{matrix}\right)log(\phi_k))\\
&=exp((T(y))_1log(\phi_1/\phi_k)+(T(y))_2log(\phi_2/\phi_k)+\dots+(T(y))_{k-1}log(\phi_{k-1}/\phi_k)+log(\phi_k))\\
&=b(y)exp(\eta^TT(y)-\alpha(\eta))
\end{align}$$

$$\eta=\begin{bmatrix}log(\phi_1/\phi_k)\\log(\phi_2/\phi_k)\\\vdots\\log(\phi_{k-1}/\phi_k)\end{bmatrix}$$

$$a(\eta)=-log(\phi_k)$$

$$b(y)=1$$

联系函数(link function)是下式（从 $i=1,\dots,k$ ）：

$$\eta_i=log\frac{\phi_i}{\phi_k}$$

其中 $\eta_k=log\frac{\phi_k}{\phi_k}=0$ ，反转连接函数，求得响应函数：

$$e^{\eta_i}=\frac{\phi_i}{\phi_k}$$

$$\phi_ke^{\eta_i}=\phi_i$$

$$\phi_k \sum_{i=1}^k e^{\eta_i} = \sum_{i=1}^k \phi_i=1$$

因此， $\phi_k=1/\begin{matrix}\sum_{i=1}^k e^{\eta_i}\end{matrix}$ ，代入 $\phi_ke^{\eta_i}=\phi_i$ 得：

$$\phi_i=\frac{e^{\eta_i}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}$$

上面从 $\eta$ 映射到 $\phi$ 的函数被称为softmax函数。

根据广义线性模型假设3， $\eta_i=\theta_i^Tx$ 其中($i=1,\dots,k$ ) 定义$\theta_k=0$ 所以 $\eta_k=\theta_k^Tx=0$ 。代入响应函数后可得给定x，y的条件分布。即模型假设。

$$\begin{align}p(y=i\mid x;\theta)
&=\phi_i\\
&=\frac{e^{\eta_i}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}\\
&=\frac{e^{\theta_i^Tx}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}
\end{align}$$

模型假设应用到 $y\in\left\{1,\dots,k\right\}$ 的分类问题时，被称为softmax回归，它是逻辑回归的推广。

假设函数：

$$\begin{align}h_{\theta}(x)
&=E[T(y)\mid x;\theta]\\
&=E\begin{bmatrix}
\begin{matrix}
1\left\{y=1\right\}\\
1\left\{y=2\right\}\\
\vdots\\
1\left\{y=k-1\right\}
\end{matrix}| x;\theta
\end{bmatrix}\\
&=\begin{bmatrix}
\phi_1\\
\phi_2\\
\vdots\\
\phi_{k-1}
\end{bmatrix}\\
&=\begin{bmatrix}
\frac{e^{\theta_1^Tx}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}\\
\frac{e^{\theta_2^Tx}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}\\
\vdots\\
\frac{e^{\theta_{k-1}^Tx}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}
\end{bmatrix}
\end{align}$$

因此，假设函数会输出 k-1 个估计概率。

**参数拟合:** 给定 $m$ 个样本的训练集，求出对数似然：

$$\begin{align}\ell(\theta)
&=\sum_{i=1}^mlogp(y^{(i)}|x^{(i)};\theta)\\
&=\sum_{i=1}^mlog\coprod_{l=1}^k \left(\frac{e^{\theta_l^Tx}}{\begin{matrix}\sum_{j=1}^k e^{\eta_j}\end{matrix}}\right)^{1\left\{y^{(i)}=l\right\}}
\end{align}$$

形如逻辑函数的 $p(y|x;\theta)=h_{\theta}(x)^y(1-h_{\theta}(x))^{(1-y)}$

用梯度下降或牛顿法求出极大似然估计的 $\theta$ 值。

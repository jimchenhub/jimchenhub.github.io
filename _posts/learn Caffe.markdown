---
layout: post
title:  "Caffe Learning Notes"
date:   2017-07-03 20:15:00 +0800
categories: Caffe
---

之前一直在学习*Transfer Learning*，看到*DAN*和*RTN*的源代码都是用Caffe写的，所以趁此机会来学习一下Caffe的源代码，为了之后能自己写出来自己的类。

本文在全局上主要参考[Caffe Source Code Analysis
](https://buptldy.github.io/2016/10/09/2016-10-09-Caffe_Code/)这篇博客。这个里面对整个初始化和训练做了一个比较高层次的简单的介绍，可以通读一下来学习基本过程。

接下来还主要参考了不同的学习笔记如下：

1. [http://blog.csdn.net/xizero00/article/details/50886829](http://blog.csdn.net/xizero00/article/details/50886829)

2. [http://blog.csdn.net/u011104550/article/details/51232667](http://blog.csdn.net/u011104550/article/details/51232667)

3. [http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html](http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html)

4. [http://alanse7en.github.io/caffedai-ma-jie-xi-1/](http://alanse7en.github.io/caffedai-ma-jie-xi-1/)

后文对这些内容进行了总结和归纳。

# Caffe 简介

一般在介绍Caffe代码结构的时候，大家都会说Caffe主要由`Blob`,`Layer`,`Net`和`Solver`这几个部分组成。

* `Blob` 主要用来表示网络中的数据，包括训练数据，网络各层自身的参数(包括权值、偏置以及它们的梯度)，网络之间传递的数据都是通过 `Blob` 来实现的，同时 `Blob` 数据也支持在 CPU 与 GPU 上存储，能够在两者之间做同步。
* `Layer` 是对神经网络中各种层的一个抽象，包括我们熟知的卷积层和下采样层，还有全连接层和各种激活函数层等等。同时每种 `Layer` 都实现了前向传播和反向传播，并通过 `Blob` 来传递数据。
* `Net` 是对整个网络的表示，由各种 `Layer` 前后连接组合而成，也是我们所构建的网络模型。
* `Solver` 定义了针对 Net 网络模型的求解方法，记录网络的训练过程，保存网络模型参数，中断并恢复网络的训练过程。自定义 `Solver` 能够实现不同的网络求解方式。

![Caffe structures](images/caffe_structure.png)

## 总体学习——通过Caffe训练LeNet来看看网络初始化和训练过程

在Caffe提供的例子里，训练LeNet网络的命令为：

```
cd $CAFFE_ROOT
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
```

其中第一个参数`build/tools/caffe`是Caffe框架的主要框架，由`tools/caffe.cpp`文件编译而来，第二个参数`train`表示是要训练网络，第三个参数是 `solver` 的 `protobuf` 描述文件。在Caffe中，网络模型的描述及其求解都是通过 `protobuf` 定义的，并不需要通过敲代码来实现。同时，模型的参数也是通过 `protobuf` 实现加载和存储，包括 CPU 与 GPU 之间的无缝切换，都是通过配置来实现的，不需要通过硬编码的方式实现。

### 网络初始化
在`caffe.cpp`中main函数之外通过`RegisterBrewFunction`这个宏在每一个实现主要功能的函数之后将这个函数的名字和其对应的函数指针添加到了`g_brew_map`中,具体分别为`train()`，`test()`，`device_query()`，`time()`这四个函数。

在运行的时候,根据传入的参数在main函数中，通过`GetBrewFunction`得到了我们需要调用的那个函数的函数指针，并完成了调用。

在我们上面所说的训练LeNet的例子中，传入的第二个参数为`train`，所以调用的函数为`caffe.cpp`中的`int train()`函数，接下来主要看这个函数的内容。在`train`函数中有下面两行代码，下面的代码定义了一个指向`Solver`的`shared_ptr`。其中主要是通过调用`SolverRegistry`这个类的静态成员函数`CreateSolver`得到一个指向`Solver`的指针来构造`shared_ptr`类型的solver。而且由于C++多态的特性，尽管solver是一个指向基类Solver类型的指针，通过solver这个智能指针来调用各个成员函数会调用到各个子类(SGDSolver等)的函数。

``` 
// caffe.cpp
// 其中输入参数solver_param就是上面所说的第三个参数：网络的模型及求解文件
shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param);
```

从上面代码可以看出，会先调用父类`Solver`的构造函数，如下所示。`Solver`类的构造函数通过`Init(param)`函数来初始化网络。

```
//solver.cpp
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),requested_early_exit_(false)
{
  Init(param);
}
```

而在`Init(paran)`函数中，又主要是通过`InitTrainNet()`和`InitTestNets()`函数分别来搭建训练网络结构和测试网络结构。

训练网络只能有一个,在`InitTrainNet()`函数中首先会设置一些基本参数，包括设置网络的状态为TRAIN，确定训练网络只有一个等，然会会通过`net_.reset(new Net<Dtype>(net_param));`这条语句新建了一个Net对象。`InitTestNets()`函数和`InitTrainNet()`函数基本类似。

上面语句新建了`Net`对象之后会调用`Net`类的构造函数，构造函数是通过`Init(param)`函数来初始化网络结构的。

在`net.cpp`里`init()`的主要内容是：其中`LayerRegistry<Dtype>::CreateLayer(layer_param)`主要是通过调用`LayerRegistry`这个类的静态成员函数`CreateLayer`得到一个指向`Layer`类的`shared_ptr`类型指针。并把每一层的指针存放在`vector<shared_ptr<Layer<Dtype> > > layers_`这个指针容器里。这里相当于根据每层的参数`layer_param`实例化了对应的各个子类层，比如`conv_layer`(卷积层)和`pooling_layer`(池化层)。实例化了各层就会调用每个层的构造函数，但每层的构造函数都没有做什么大的设置。

`init()`函数主要又四个部分组成：

* AppendBottom：设置每一层的输入数据
* AppendTop：设置每一层的输出数据
* layers_[layer_id]->SetUp：对上面设置的输入输出数据计算分配空间，并设置每层的可学习参数(权值和偏置)
* AppendParam：对上面申请的可学习参数进行设置，主要包括学习率和正则率等。

```
//net.cpp Init()
for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {//param是网络参数，layer_size()返回网络拥有的层数
    const LayerParameter& layer_param = param.layer(layer_id);//获取当前layer的参数
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));//根据参数实例化layer


//下面的两个for循环将此layer的bottom blob的指针和top blob的指针放入bottom_vecs_和top_vecs_,bottom blob和top blob的实例全都存放在blobs_中。相邻的两层，前一层的top blob是后一层的bottom blob，所以blobs_的同一个blob既可能是bottom blob，也可能使top blob。
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();++bottom_id) {
       const int blob_id=AppendBottom(param,layer_id,bottom_id,&available_blobs,&blob_name_to_idx);
    }

    for (int top_id = 0; top_id < num_top; ++top_id) {
       AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }

// 调用layer类的Setup函数进行初始化，输入参数：每个layer的输入blobs以及输出blobs,为每个blob设置大小
layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);

//接下来的工作是将每层的parameter的指针塞进params_，尤其是learnable_params_。
   const int num_param_blobs = layers_[layer_id]->blobs().size();
   for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
       AppendParam(param, layer_id, param_id);
       //AppendParam负责具体的dirtywork
    }


    }
```

经过上面的过程，`Net`类的初始化工作基本就完成了。总体的流程大概就是新建一个`Solver`对象，然后调用`Solver`类的构造函数，然后在`Solver`的构造函数中又会新建`Net`类实例，在`Net`类的构造函数中又会新建各个`Layer`的实例,一直具体到设置每个`Blob`,大概就介绍完了网络初始化的工作。

### 训练过程
完成初始化之后，就可以开始对网络经行训练了，开始训练的代码如下所示，指向Solver类的指针solver开始调用Solver类的成员函数Solve()，名称比较绕啊。

```
// 开始优化
solver->Solve();
```

`Solve`函数其实主要就是调用了`Solver`的另一个成员函数`Step()`来完成实际的迭代训练过程。

```
//solver.cpp
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  ...
  int start_iter = iter_;
  ...
  // 然后调用了'Step'函数，这个函数执行了实际的逐步的迭代过程
  Step(param_.max_iter() - iter_);
  ...
  LOG(INFO) << "Optimization Done.";
}
```

顺着来看看这个`Step()`函数的主要代码,首先是一个大循环设置了总的迭代次数，在每次迭代中训练iter_size x batch_size个样本，这个设置是为了在GPU的显存不够的时候使用，比如我本来想把batch_size设置为128，iter_size是默认为1的，但是会`out_of_memory`，借助这个方法，可以设置batch_size=32，iter_size=4，那实际上每次迭代还是处理了128个数据。

```
//solver.cpp
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  ...
  //迭代
  while (iter_ < stop_iter) {
    ...
    // iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size，
    // 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size，这个loss是通过调用`Net::ForwardBackward`函数得到的
    // accumulate gradients over `iter_size` x `batch_size` instances
    for (int i = 0; i < param_.iter_size(); ++i) {
    /*
     * 调用了Net中的代码，主要完成了前向后向的计算，
     * 前向用于计算模型的最终输出和Loss，后向用于
     * 计算每一层网络和参数的梯度。
     */
      loss += net_->ForwardBackward();
    }

    ...

    /*
     * 这个函数主要做Loss的平滑。由于Caffe的训练方式是SGD，我们无法把所有的数据同时
     * 放入模型进行训练，那么部分数据产生的Loss就可能会和全样本的平均Loss不同，在必要
     * 时候将Loss和历史过程中更新的Loss求平均就可以减少Loss的震荡问题。
     */
    UpdateSmoothedLoss(loss, start_iter, average_loss);


    ...
    // 执行梯度的更新，这个函数在基类`Solver`中没有实现，会调用每个子类自己的实现
    //，后面具体分析`SGDSolver`的实现
    ApplyUpdate();

    // 迭代次数加1
    ++iter_;
    ...

  }
}
```

上面`Step()`函数主要分为三部分：

```
loss += net_->ForwardBackward();
```

这行代码通过`Net`类的`net_`指针调用其成员函数`ForwardBackward()`，其代码如下所示,分别调用了成员函数`Forward(&loss)`和成员函数`Backward()`来进行前向传播和反向传播。

```
// net.hpp
// 进行一次正向传播，一次反向传播
Dtype ForwardBackward() {
  Dtype loss;
  Forward(&loss);
  Backward();
  return loss;
}
```

前面的`Forward(&loss)`函数最终会执行到下面一段代码,`Net`类的`Forward()`函数会对网络中的每一层执行`Layer`类的成员函数`Forward()`，而具体的每一层`Layer`的派生类会重写`Forward()`函数来实现不同层的前向计算功能。上面的`Backward()`反向求导函数也和`Forward()`类似，调用不同层的`Backward()`函数来计算每层的梯度。

```
//net.cpp
for (int i = start; i <= end; ++i) {
// 对每一层进行前向计算，返回每层的loss，其实只有最后一层loss不为0
  Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
  loss += layer_loss;
  if (debug_info_) { ForwardDebugInfo(i); }
}
```

```
UpdateSmoothedLoss();
```

这个函数主要做`Loss`的平滑。由于Caffe的训练方式是SGD，我们无法把所有的数据同时放入模型进行训练，那么部分数据产生的`Loss`就可能会和全样本的平均`Loss`不同，在必要时候将`Loss`和历史过程中更新的`Loss`求平均就可以减少`Loss`的震荡问题

```
ApplyUpdate();
```

这个函数是`Solver`类的纯虚函数，需要派生类来实现，比如SGDSolver类实现的`ApplyUpdate()`;函数如下，主要内容包括：设置参数的学习率；对梯度进行Normalize；对反向求导得到的梯度添加正则项的梯度；最后根据SGD算法计算最终的梯度；最后的最后把计算得到的最终梯度对权值进行更新。

```
template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());

  // GetLearningRate根据设置的lr_policy来计算当前迭代的learning rate的值
  Dtype rate = GetLearningRate();

  // 判断是否需要输出当前的learning rate
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }

  // 避免梯度爆炸，如果梯度的二范数超过了某个数值则进行scale操作，将梯度减小
  ClipGradients();

  // 对所有可更新的网络参数进行操作
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
	// 将第param_id个参数的梯度除以iter_size，
	// 这一步的作用是保证实际的batch_size=iter_size*设置的batch_size
    Normalize(param_id);

    // 将正则化部分的梯度降入到每个参数的梯度中
    Regularize(param_id);

    // 计算SGD算法的梯度(momentum等)
    ComputeUpdateValue(param_id, rate);
  }
  // 调用`Net::Update`更新所有的参数
  this->net_->Update();
}
```

等进行了所有的循环，网络的训练也算是完成了。上面大概说了下使用Caffe进行网络训练时网络初始化以及前向传播、反向传播、梯度更新的过程，其中省略了大量的细节。上面还有很多东西都没提到，比如说Caffe中`Layer`派生类的注册及各个具体层前向反向的实现、`Solver`派生类的注册、网络结构的读取、模型的保存等等大量内容。

## 模块学习——Layer

![Layer structures](images/layer_structure.png)

* `Activation/NeuronLayer`类 定义于`neuron_layers.hpp`中，其派生类主要是元素级别的运算（比如Dropout运算，激活函数ReLu，Sigmoid等），运算均为同址计算（in-place computation，返回值覆盖原值而占用新的内存）。
* `LossLayer`类 定义于`loss_layers.hpp`中，其派生类会产生loss，只有这些层能够产生loss。
* `DataLayer` 定义于`data_layer.hpp`中，作为网络的最底层，主要实现数据格式的转换。
* `Vision` Layer 定义于`vision_layers.hpp`，实现特征表达功能，更具体地说包含卷积操作，Pooling操作，他们基本都会产生新的内存占用（Pooling相对较小）。
* `Common Layers` 定义于`common_layers.hpp`，Caffe提供了单个层与多个层的连接，并在这个头文件中声明。这里还包括了常用的全连接层`InnerProductLayer`类。


Layer层的主要的参数和成员变量:

```
 /** The protobuf that stores the layer parameters */
 // 层说明参数，从protocal buffers格式的网络结构说明文件中读取
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  // 层状态，参与网络的训练还是测试
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  // 层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob. */
  // 标志每个top blob是否需要计算反向传递的梯度值
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
   // 每一个top blob中是否有非零的权值
  vector<Dtype> loss_;
```

初始化函数SetUp()

```
/**
   * @brief 实现每个layer对象的setup函数
   * @param bottom 
   * 层的输入数据，blob中的存储空间已申请
   * @param top
   * 层的输出数据，blob对象以构造但是其中的存储空间未申请，
   * 具体空间大小需根据bottom blob大小和layer_param_共同决定，具体在Reshape函数现实

   * 1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
   * 2. 调用LayerSetUp函数初始化特殊的层，每个Layer子类需重写这个函数完成定制的初始化
   * 3. 调用Reshape函数为top blob分配合适大小的存储空间
   * 4. 为每个top blob设置损失权重乘子，非LossLayer为的top blob其值为零
   *
   * 此方法非虚函数，不用重写，模式固定
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }
```

每个子类Layer必须重写的初始化函数LayerSetUp和Reshape函数，其中Reshape函数完成top blob形状的设置并为其分配存储空间。

```
 /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *        定制初始化，每个子类layer必须实现此虚函数
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   *     输入blob, 数据成员data_和diff_存储了相关数据
   * @param top
   *     the allocated but unshaped output blobs
   *     输出blob, blob对象已构造但数据成员的空间尚未申请
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数，
   * 调用Reshape函数申请top blob的存储空间
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
      
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
```

前向传播函数Forward和反向传播函数Backward。这两个函数非虚函数，它们内部会调用如下虚函数完成数据前向传递和误差反向传播，根据执行环境的不同每个子类Layer必须重写CPU和GPU版本

```
  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }
```

## 模块学习——Protocol Buffer
除了清晰的代码结构，让Caffe变得易用更应该归功于Google Protocol Buffer的使用。Google Protocol Buffer是Google开发的一个用于serializing结构化数据的开源工具:

> Protocol buffers are a language-neutral, platform-neutral extensible mechanism for serializing structured data.

Caffe使用这个工具来定义`Solver`和`Net`，以及`Net`中每一个`layer`的参数。这使得只是想使用Caffe目前支持的`Layer`(已经非常丰富了)来做一些实验或者demo的用户可以不去和代码打交道，只需要在`*.prototxt`文件中描述自己的`Solver`和`Net`即可，再通过Caffe提供的command line interfaces就可以完成模型的train, finetune, test等功能。[1]

**Protocol Buffer的学习** [Protocol Buffer Basics: C++](https://developers.google.com/protocol-buffers/docs/cpptutorial#why-use-protocol-buffers)

#### Caffe中的应用
caffe当中的使用可以见 `caffe/src/caffe/proto/caffe.proto`  
Reference: [Protocol Buffer in Caffe](http://alanse7en.github.io/caffedai-ma-jie-xi-2/)

prototxt的编写可以直接参考官方文档，可以参考的地方也很多，编写时也有一个可视化的工具来帮梦，地址入下：
[Quick Start — Netscope](http://ethereon.github.io/netscope/quickstart.html)

### Reference
[1] http://alanse7en.github.io/caffedai-ma-jie-xi-1/ 

## 模块学习——Command Line Interfaces
### Google Flags
Caffe的Command Line Interfaces一共提供了四个功能：`train`, `test`, `time`, `device_query`，而Interfaces的输入除了这四种功能还可以输入诸如`-solver`, `-weights`, `-snapshot`, `-gpu`等参数。 *这些参数的解析是通过Google Flags这个工具来完成的* 。[1]

解析这些标志的代码在caffe.cpp中的main()中调用了/CAFFE_ROOT/src/common.cpp中的GlobalInit(&argc, &argv)函数：

```
void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}
```

### Register Brew Function
Caffe在Command Line Interfaces中一共提供了4种功能:`train`/`test`/`time`/`device_query`，分别对应着四个函数，这四个函数的调用是通过一个叫做`g_brew_map`的全局变量来完成的：

```
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
```

`g_brew_map`是一个key为`string`类型，value为`BrewFunction`类型的一个map类型的全局变量，`BrewFunction`是一个函数指针类型，指向的是参数为空，返回值为int的函数，也就是`train`/`test`/`time`/`device_query`这四个函数的类型。在`train`等四个函数实现的后面都紧跟着这样一句宏的调用：RegisterBrewFunction(train);

总结一下：RegisterBrewFunction这个宏在每一个实现主要功能的函数之后将这个函数的名字和其对应的函数指针添加到了`g_brew_map`中，然后在main函数中，通过`GetBrewFunction`得到了我们需要调用的那个函数的函数指针，并完成了调用。

具体可以参考 reference [1] [Caffe代码解析(3) – Xuesong’s Blog](http://alanse7en.github.io/caffedai-ma-3/)

### train() 函数

```
 CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
    << "Give a snapshot to resume training or weights to finetune "
    "but not both.";
```

这段代码的第一行使用了`glog`的`CHECK_GT`宏（含义为check greater than），检查`FLAGS_solver`的size是否大于0，如果小于或等于0则输出提示：”Need a solver definition to train”。`FLAGS_solver`是最开始通过`DEFINE_string`定义的标志，如果我们希望训练一个模型，那么自然应该应该提供对应的solver定义文件的路径，这一句话正是在确保我们提供了这样的路径。这样的检查语句在后续的代码中会经常出现，将不再一一详细解释，如果有不清楚含义的`glog`宏可以去看看文档。 与第一行代码类似，第二行代码是确保用户没有同时提供 snapshot 和 weights 参数，这两个参数都是继续之前的训练或者进行fine-tuning的，如果同时指明了这两个标志，则不知道到底应该从哪个路径的文件去读入模型的相关参数更为合适。

然后出现了 `SolverParameter solver_param` 的声明和解析的代码：
```
caffe::SolverParameter solver_param;
caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
```

`SolverParameter`是通过Google Protocol Buffer自动生成的一个类，而具体的解析函数将在下一部分具体解释。

接下来这一部分的代码是根据用户的设置来选择Caffe工作的模式（GPU或CPU）以及使用哪些GPU(caffe已经支持了多GPU同时工作！具体参考： [Caffe | Interfaces](http://caffe.berkeleyvision.org/tutorial/interfaces.html)

```
  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
```

首先是判断用户在Command Line中是否输入了gpu相关的参数，如果没有(FLAGS\_gpu.size()==0)但是用户在solver的prototxt定义中提供了相关的参数，那就把相关的参数放到FLAGS\_gpu中，如果用户仅仅是选择了在solver的prototxt定义中选择了GPU模式，但是没有指明具体的gpu\_id，那么就默认设置为0。

接下来的代码则通过一个get\_gpus的函数，将存放在FLAGS\_gpu中的string转成了一个vector，并完成了具体的设置。[1]

### References
[1] http://alanse7en.github.io/caffedai-ma-3/

## 模块学习——Blob类

主要参考自 [楼燚(yì)航的blog——Caffe源码解析1：Blob](http://www.cnblogs.com/louyihang-loves-baiyan/)

下面会对具体的代码进行分析，但是如果对整体代码有想参考的部分，可以参考另外一篇博客：[caffe 中 BLOB的实现
](http://blog.csdn.net/xizero00/article/details/50886829)，其中对整个代码部分进行了注释，可以参考学习。

首先提一下`explicit`关键字的作用是禁止单参数构造函数的隐式转换。`iniline`主要是将代码进行复制，扩充，会使代码总量上升，好处就是可以节省调用的开销，能提高执行效率。

实际上`Blob`包含了三类数据

1. data，前向传播所用到的数据
1. diff，反向传播所用到的数据
1. shape，解释data和diff的shape数据


### 主要变量
```
shared_ptr<SyncedMemory> data_;
shared_ptr<SyncedMemory> diff_;
shared_ptr<SyncedMemory> shape_data_;
vector<int> shape_;
int count_;
int capacity_;
```

Blob只是一个基本的数据结构，因此内部的变量相对较少，首先是 `data_` 指针，指针类型是 `shared_ptr` ，属于 `boost` 库的一个智能指针，这一部分主要用来申请内存存储 `data` ， `data` 主要是正向传播的时候用的。同理， `diff_` 主要用来存储偏差，update data ， `shape_data` 和 `shape_` 都是存储Blob的形状，一个是老版本一个是新版本。 `count` 表示Blob中的元素个数，也就是 `个数*通道数*高度*宽度` , `capacity` 表示当前的元素个数，因为`Blob`可能会reshape。

### 主要函数
#### Reshape()
```
  void Reshape(const int num, const int channels, const int height, const int width);
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
```
Blob中除了基础的构造函数，还有大量的 `Reshape` 函数， `Reshape` 函数在Layer中的 `reshape` 或者 `forward` 操作中来adjust dimension。同时在改变`Blob`大小时，内存将会被重新分配，如果内存大小不够了，并且额外的内存将不会被释放。对input的blob进行reshape,如果立马调用 `Net::Backward` 是会出错的，因为reshape之后，要么 `Net::forward` 或者 `Net::Reshape` 就会被调用来将新的input shape传播到高层。

#### count()
```
inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }
  inline int count(int start_axis, int end_axis) const {
    ...
  }
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }
```
Blob类里面有重载很多个count()函数，主要还是为了统计Blob的容量（volume），或者是某一片（slice），从某个axis到具体某个axis的shape乘积。 

**并且Blob的Index是可以从负坐标开始读的，这一点跟Python好像**

对于Blob中的4个基本变量 `num, channel, height, width` 可以直接通过shape(0),shape(1),shape(2),shape(3)来访问。

#### Data & Diff
```
inline Dtype data_at(const int n, const int c, const int h, const int w)
inline Dtype diff_at(const int n, const int c, const int h, const int w)
inline Dtype data_at(const vector<int>& index)
inline Dtype diff_at(const vector<int>& index)
inline const shared_ptr<SyncedMemory>& data()
inline const shared_ptr<SyncedMemory>& diff()
```
这一部分函数主要通过给定的位置访问数据，根据位置计算与数据起始的偏差offset，在通过cpu_data\*指针获得地址。下面几个函数都是获得
```
const Dtype* cpu_data() const;
void set_cpu_data(Dtype* data);
const int* gpu_shape() const;
const Dtype* gpu_data() const;
const Dtype* cpu_diff() const;
const Dtype* gpu_diff() const;
Dtype* mutable_cpu_data();
Dtype* mutable_gpu_data();
Dtype* mutable_cpu_diff();
Dtype* mutable_gpu_diff();
```
可以看到这里有data和diff两类数据，**而这个diff就是我们所熟知的偏差，前者主要存储前向传递的数据，而后者存储的是反向传播中的梯度**

#### Update()
```
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}
```
这个里面核心的一句话就是

```
caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));  
```

这句话是在 `include/caffe/util/math_function.cpp` 文件中

```
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }
```

**可以看到这一段调用了cblas库的方法**。实现的功能是 

> Y=alpha∗X+beta∗Y

也就是blob里面的data部分减去diff部分

#### norm

```
Dtype asum_data() const;//计算data的L1范数
Dtype asum_diff() const;//计算diff的L1范数
Dtype sumsq_data() const;//计算data的L2范数
Dtype sumsq_diff() const;//计算diff的L2范数
void scale_data(Dtype scale_factor);//将data部分乘以一个因子
void scale_diff(Dtype scale_factor);//将diff部分乘一个因子
```


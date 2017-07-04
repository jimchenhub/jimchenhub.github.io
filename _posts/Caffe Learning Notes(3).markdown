---
layout: post
title:  "Caffe Learning Notes(3)"
date:   2017-07-04 17:00:00 +0800
categories: Caffe
---

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

## 模块学习——SyncedMemory
看到SyncedMem就知道，这是在做内存同步的操作。这类个类的代码比较少，但是作用是非常明显的。文件对应着syncedmem.hpp,着syncedmem.cpp

首先是两个全局的内联函数。如果机器是支持GPU的并且安装了cuda，通过cudaMallocHost分配的host memory将会被pinned，**这里我谷歌了一下，pinned的意思就是内存不会被paged out，我们知道内存里面是由页作为基本的管理单元。分配的内存可以常驻在内存空间中对效率是有帮助的，空间不会被别的进程所抢占。**同样如果内存越大，能被分配的Pinned内存自然也越大。**还有一点是，对于单一的GPU而言提升并不会太显著，但是对于多个GPU的并行而言可以显著提高稳定性。** [1]

### Inference
[1] http://www.cnblogs.com/louyihang-loves-baiyan/p/5150554.html


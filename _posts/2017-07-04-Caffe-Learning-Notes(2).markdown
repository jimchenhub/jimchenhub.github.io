---
layout: post
title:  "Caffe Learning Notes(2)"
date:   2017-07-04 12:00:00 +0800
categories: Caffe
---

# 模块学习——Protocol Buffer
除了清晰的代码结构，让Caffe变得易用更应该归功于Google Protocol Buffer的使用。Google Protocol Buffer是Google开发的一个用于serializing结构化数据的开源工具:

> Protocol buffers are a language-neutral, platform-neutral extensible mechanism for serializing structured data.

Caffe使用这个工具来定义`Solver`和`Net`，以及`Net`中每一个`layer`的参数。这使得只是想使用Caffe目前支持的`Layer`(已经非常丰富了)来做一些实验或者demo的用户可以不去和代码打交道，只需要在`*.prototxt`文件中描述自己的`Solver`和`Net`即可，再通过Caffe提供的command line interfaces就可以完成模型的train, finetune, test等功能。[1]

**Protocol Buffer的学习** [Protocol Buffer Basics: C++](https://developers.google.com/protocol-buffers/docs/cpptutorial#why-use-protocol-buffers)

### Caffe中的应用
caffe当中的使用可以见 `caffe/src/caffe/proto/caffe.proto`  
Reference: [Protocol Buffer in Caffe](http://alanse7en.github.io/caffedai-ma-jie-xi-2/)

prototxt的编写可以直接参考官方文档，可以参考的地方也很多，编写时也有一个可视化的工具来帮梦，地址入下：
[Quick Start — Netscope](http://ethereon.github.io/netscope/quickstart.html)

## Reference
[1] http://alanse7en.github.io/caffedai-ma-jie-xi-1/ 

# 模块学习——Command Line Interfaces
## Google Flags
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

## Register Brew Function
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

## train() 函数

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

## References
[1] http://alanse7en.github.io/caffedai-ma-3/

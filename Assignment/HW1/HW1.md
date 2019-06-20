## 17级并行与分布式计算

#### HW1 17341137 宋震鹏

1. 在下载大批量小文件时，选用并发的下载方式会大幅提升效率。此处利用了任务并行(Task-parallelism)的思想。举例进行具体解释：计划下载10K个20kb以内的文件，下载器原本执行串行下载，需要时间较长，若修改下载器下载方式，同时进行多个文件的并发式下载，则用时大幅降低。

   

2. 

   1. Digress：简单地说，计算机内乘法操作以“左移”操作和“加法”操作组成，因此一次乘法操作比加法操作是较慢的。

   2. Origin ALU：$0.23\times8+0.77\times Rest = 6.7$

      Improved ALU：$0.23\times6+0.77\times Rest = ALU_{improved}$

      By these above, $ALU_{improved}=6.24$

      Origin：$0.47\times6.7+0.19\times7.9+0.20\times5.0+0.14\times7.1=6.644​$

      Improved：$0.47\times6.24+0.19\times7.9+0.20\times5.0+0.14\times7.1=6.4278$

      It’s improved by $\eta =\frac{6.644}{6.4278} = 1.0336$.

---

17341137 宋震鹏 19/3/6 0:18


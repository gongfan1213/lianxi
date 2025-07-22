# 纯物理
linux部署的方式

v m wa re虚拟机+centos（ios经箱）=得到一个可以使用的linux系统的

linux机器上进行部署如上的软件

安装

修改配置文件

部署非常慢

物理服务器，真实购买的一台硬件机器


# 虚拟化技术


一台机器上运行多个操作系统的，64gb的服务器的，agents

一个物理机器可以部署多个app，
每一个app运行一个，资源浪费的


虚拟机的模版功能，vmware提供了模版的克隆的，vmware2的

虚拟化有局限的，每个虚拟机是一个完整的操作系统的，要分配系统的资源，虚拟机的多道一定程度的，操作系统本身的资源也就消耗殆尽的，


# 容器技术的

虚拟化技术

云计算的

把计算的工作放到云上执行的

阿里云购买rds的数据库的服务，不需要自己搭建数据库，做数据库的高可用等，用的全部都是线上的云数据库的场景的

阿里云大量的虚拟化的手段的，配置越高价格越高的，网络存储系统的，各种云平台来完成的，

虚拟化技术的，每一个app独立运行在一个vm里的

虚拟机工具的巨头的

vmware workstation属于个人学习使用的

企业版的虚拟化的，vmware exsi虚拟化工具，高性能服务结合的

kvm和容器技术的，

vm工具主要是用于windows平台的

linux下的虚拟机的工具，kvm工具的，创建虚拟机的，安装各种的系统

虚拟化

各种实体资源进行抽象的，组合成一个可以为多个假的不存在的机器的，

c pu,memory,disk,network,linux的各种的应用程序的

虚拟化的技术的，

vmware

二进制的翻译技术

vmware，vmware当中的系统，将虚拟机当中要执行的指令翻译称为交给对应的乌力吉执行

全虚拟化的


xen

半虚拟化的

满足虚拟机的操作系统的需要，不影响到物理的计算机的

硬件辅助虚拟化的VT/AMD-v

虚拟化的模式的，

VT

二进制翻译

cpu辅助虚拟化的


kvm0qemu

for kernel-based-

hadrware

linux kernel

硬件辅助虚拟化的

独立的第三方软件的，kvm继承在linux当中，

KVM，vcpu,vmem

qemu-kvm

dock er

完整的操作系统的，占用资源很大的，

容器技术的，翻译二进制的全虚拟化技术的，完整的操作系统，虚拟出来一台完整的计算机的，对宿主机器一定的损耗的，完整的计算机的，物理硬件的

仅仅使用多个应用程序，希望，他们之间互相独立的

物理机
yu m install mysql -5.7 -y

另外一个需求需要mysql5.8的

造成冲突了，希望是一个场景的，利用docker希望得到如下的场景的

centos宿主机器

docker技术的

容器1，容器2，容器3

每一个容器相互独立的，实现了环境的隔离的，

单独的环境的，虚拟出一个计算机的成本高的，虚拟出一个隔离的运行环境的成本低的

虚拟出一个隔离的环境的。对于资源的充分利用的

# docker容器技术的

虚拟化平台当中,openstack

所有的应用系统，运行在公司内部的虚拟机平台当中，有一个平台网站，可以非常方便的管理公司的服务器资源的，进行快捷的虚拟机的创建，删除，扩容，

这个平台通过linux技术的，openstack搭建的
vmware workstation图形化工具

创建虚拟机的，运行完整的操作系统，

随着应用不断的增加的，

云主机越来越多的

利用硬件资源实现云服务的

# 容器技术

Golang语言开发的，基于linux内核的Cgroups，Namespace以及UnionFs等技术，对进程进行封装隔离，属于操作系统层面的

centos宿主几的，mysql5.7

0-65535每一个容器相互独立，清凉级别的环境隔离系统的

由于隔离的进程独立于宿主机和其他的隔离进程也称之为容器的

利用docker容器，

开发同学程序员，这是程序员小雨的电脑的，程序开发完毕的

测试工程师，小李的，程序代码，将代码放到测试服务器当中

生产服务器，线上服务器centos，线上去运行的，测试代码，安装以来上传代码，修改配置的，尝试启动的，

生产服务器，线上服务器centors7

交给测试，再交给运维的，

认为的操作，必然可能会存在错误的操作的，后段交给运维的，容器技术出现之后，后段的代码的，本地环境，测试环境，和生产环境，有了docker容器周部署变得非常简单了，

依赖关系，可能会依赖的关系的，程序员在代码写好之后，还会开发dockerfile脚本的，将代码和环境依赖全部打包成为一个镜像文件的，整个圈圈全部拷贝到测试环境，生产环境的，不需要再像以前一样各种配置各种询问的

容器之后打包成一个镜像的docker run直接运行了，放到生产环境当中的

虚拟机是虚拟出一套硬件，制定系统镜像，然后装系统，最终可以使用，再改系统上在运行所需要的应用程序的

kvm创建虚拟机的时候，制定较少的cpu，内存，硬件等资源，虚拟机性能较低的

虚拟机：每一个虚拟应用程序不仅包括应用程序和必要的二进制文件等等

容器内的应用程序直接运行在宿主几的内核上的
yu m install docker

容器内的程序就好比直接运行在宿主几上，能够使用数足迹最大的硬件资源的

但是他们又是相互隔离的

虚拟机的，1g内核的50g磁盘的里面运行的应用程序的，效率必然地下的

服务器，宿主几操作程序，docker引擎的，

## 容器和kvm对比的
容器能够提供宿主几的性能，但是kvm虚拟机是分配宿主几的硬件资源的，性能比较弱的

同样配置的宿主机，最多可以启动10个虚拟机的话，可以启动100+的容器数量的

启动一个kvm虚拟机，必须要要有一个完整的开机流程，花费时间比较长，但是启动一个容器只需要1s

kvm需要硬件cpu的虚拟化支持，但是容器不需要的

启动一个kvm虚拟机必须要有一个完整的开机流程，

kvm需要硬件cpu的虚拟化的支持

更佳高校的利用系统资源，

传递镜像文件就可以了，windows下开发的，利用docker的镜像部署，测试服务器的，开发测试服务器，限制测试服务器和生产服务器一致的

跨平台，跨系统的，系统依赖冲突，导致难以部署等的bug

除了内核以外的完整的运行的环境，确保了应用环境的一致性的

更轻松的迁移

docker的使用情况

应用程序非常之多，

## docker的安装部署的

c/s架构的

server docker daemon

api接口的，rest api
docker客户端的命令行的，

网络，容器，镜像，数据卷的，

容器有一定的网络，外部的应用访问容器的，自己的网络，镜像images，跑的容器镜像生成容器的，


镜像

容器网络存储的

<img width="311" height="194" alt="image" src="https://github.com/user-attachments/assets/8cdd2b76-1c76-418f-9615-d6748ed88e04" />


客户端，基于dockerfile构建镜像，自己构建镜像或者下载别人的家ing想，启动docker容器的


镜像，容器的，服务器的，docker主机的

# docker安装部署

image镜像，构建容器，程序运行苏需要的环境打包成文件

container
容器，应用程序保存在容器当中

镜像仓库：保存镜像文件，提供上传，下载镜像的，

dockerfile部署项目的操作写成一个部署的脚本的

构建容器的过程就是运行镜像m
conatiner就是容器的镜像的运行实例的，

docker的安装的，镜像仓库

安装docker，

提前准备好一个宿主几的，安装docker去使用的

虚拟化+容器的技术的

基于虚拟化的手段创建大量的虚拟机的，容器的

可以是笔记本公司的服务器的，

机器的基础之上的，上层的虚拟机的

guestos，

虚拟机的

yu m makecahce

systectl disable firewalid

基础的环境配置的

centos7平台不低于3.10的

sysctl -p /etc/sysctl.d/docker.conf

添加完毕之后的，提前配置好的yun的仓库的

阿里云自带的仓库的，yumlist docker-ce--showduplicates | soirt -r

可以直接yum安装docker的

yum list

配置docker加速器的

获取镜像文件的，使用最终的容器必须要有对应的镜像的,docker.pub下载
systemctl restart docker

docker ps

docker images

cs架构的，是否正确启动的

# dokcer的实际的操作使用的

获取镜像

运行镜像生成容器的

nginx，web服务器的，运行一个80端口的网站的

开启服务器

2.服务器上安装好运行nginx所需要的依赖关系的

安装nginx，yum install nginx -y
修改nginx配置文件

客户端去访问nignx

用docker获取nginx怎么做？

1.获取镜像，从你配置的docker镜像站当中

docker pull nginx

先搜索一下镜像文件是否存在的，docker search nginx


dock er pull nginx

latest

docker images ls
查看本地的镜像有哪些？

docker rumi d la364db548ctd

删除镜像的命令

docker rmi 镜像id

Digest，Status
dock er image ls

运行镜像运行处具体的容器，然后这个容器当中跑出一个nginx服务的

docker run参数 镜像的名字/id

-d 后台的运行容器的

-p 80:90 端口的映射，宿主几端口“容器内的端口的

你访问宿主几的端口的

镜像的管理

容器的管理

dock er ps的
此时访问服务端的端口的，nginx，
nets ta te -tunlp

/docker

停止容器，查看结果
dock er stop.容器id

m诶呦存活当中的容器的

dock er start容器id

跑到容器当中docker start/stop就可以的



dock er build构建dock erfile 生成镜像， dock儿images

localdocker instance存储在机器的本地的，



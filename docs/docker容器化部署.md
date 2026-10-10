# Docker容器化部署

## 快速开始

1. 构建镜像

```sh
make image
```

2. 运行容器

```sh
# 运行调度器
docker run -d --name scheduler vcluster-system scheduler -p 50050:50050

# 运行工作节点
docker run -d --name worker vcluster-system worker -p 50051:50051

# 运行客户端
docker run -d --name client vcluster-system client -p 50070:50070
```

3. 停止容器

```sh
docker stop <component_name>
```

4. 删除容器

```sh
docker rm <component_name>
```

5. 删除镜像

```sh
docker rmi vcluster-system
```

## TODO

1. 集群配置读取，需要完善config中的builder模块

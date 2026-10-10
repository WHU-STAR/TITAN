# Docker 部署

## 构建镜像

```sh
make image
```

镜像名是 `titan`。构建前会创建空的 `data`、`log`、`pids` 目录，避免镜像里缺少这些路径。

## 运行

端口参数要写在镜像名前面：

```sh
docker run -d --name scheduler -p 50050:50050 titan scheduler
docker run -d --name worker -p 50051:50051 titan worker
docker run -d --name client -p 50070:50070 titan client
```

配置通过环境变量 `TITAN_CONFIG_BUILDER` 选择 `runconfig.yaml` 里的条目。

## 停止和删除

```sh
docker stop scheduler worker client
docker rm scheduler worker client
docker rmi titan
```

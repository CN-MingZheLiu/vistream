# VIStream · GitHub Pages

VIStream 研究项目网站的独立 GitHub Pages 源代码，包含视频、论文、实验结果与交互轨迹演示。

网站地址：[https://cn-mingzheliu.github.io/vistream/](https://cn-mingzheliu.github.io/vistream/)

源码仓库：[CN-MingZheLiu/vistream](https://github.com/CN-MingZheLiu/vistream)。发布状态以仓库的 Actions 运行结果为准。

## 部署

1. 在你自己的 GitHub 账号下创建公开仓库 `vistream`，将这个目录的内容提交到 `main` 分支。需要包含隐藏文件 `.github/workflows/pages.yml` 和 `.nojekyll`。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 在 **Actions** 中运行 **Deploy VIStream to GitHub Pages**；之后向 `main` 推送更改会自动发布。
4. 等待部署成功，在该工作流或仓库 Pages 设置中打开 GitHub 给出的正式地址。

工作流使用 GitHub 官方 Pages Actions。部署时，`configure-pages` 返回真实 `base_url`，构建脚本据此设置 canonical、Open Graph URL 和预览图的绝对 URL。网站中的图片、视频、脚本、论文和数据引用均采用相对路径，可以直接部署在 `/vistream/` 或其他项目路径下。

## 本地预览和检查

在本目录运行：

```sh
python3 scripts/verify_site.py
python3 -m http.server 8000
```

然后打开 `http://localhost:8000/`。请使用 HTTP 预览；直接双击 HTML 文件会导致浏览器限制 JSON 数据的加载。

构建脚本只发布 `index.html`、`app.js`、`style.css`、`.nojekyll` 和 `assets/`。README、工作流、QA 报告及辅助脚本不会出现在部署产物中。

## 内容与证据

- 完整保留原网站的视频、论文、图片、字幕、交互轨迹数据、结果表和复现说明。
- 实验文件没有重新计算或修改；本次仅迁移托管方式并更新部署元数据。
- 网页上的轨迹展示是已记录结果的回放。完整的模型权重与离线推理环境仍属于另行交付的复现包，本网站源代码并不包含那份 11 GB 复现包。
- `QA/STATIC_VALIDATION.json` 记录静态链接检查；`QA/ASSET_PRESERVATION.json` 记录与原网站文件逐项比较的 SHA256；`QA/BUILD_PROJECT_PATH_TEST.json` 记录项目子路径构建检查。

此目录无需 npm、第三方 CDN、访问令牌或服务端。GitHub 工作流使用平台提供的权限，不需要把个人密钥写入代码。

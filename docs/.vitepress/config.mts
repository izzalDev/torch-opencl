import { defineConfig } from "@lando/vitepress-theme-default-plus/config";
import { pyproject, getVersion, injectDynamicTagline } from "./metadata";

const currentVersion = getVersion();

export default defineConfig({
  base: process.env.BASE_PATH || "/",
  outDir: "./dist",
  title: "Torch OpenCL",
  description: pyproject.project.description,

  // Dynamic page transformation
  transformPageData: injectDynamicTagline,

  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Examples", link: "/markdown-examples" },
      {
        text: currentVersion,
        items: [
          { text: "Changelog", link: "/CHANGELOG" },
          { text: "Contributing", link: "/CONTRIBUTING" },
        ],
      },
    ],
    sidebar: [
      {
        text: "Architecture",
        items: [
          { text: "Overview", link: "/architecture/" },
          { text: "System Overview", link: "/architecture/system-overview" },
          { text: "Runtime Flow", link: "/architecture/runtime-flow" },
          { text: "Runtime Layer", link: "/architecture/runtime-layer" },
          { text: "Allocator Design", link: "/architecture/allocator-design" },
          { text: "Device Allocator", link: "/runtime/device-allocator" },
          {
            text: "Dispatcher Integration",
            link: "/architecture/dispatcher-integration",
          },
          { text: "Build Pipeline", link: "/architecture/build-pipeline" },
          { text: "Testing Strategy", link: "/architecture/testing-strategy" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/izzaldev/torch-opencl" },
    ],
    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © 2026-present izzalDev",
    },
  },
});

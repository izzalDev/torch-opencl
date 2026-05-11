import { withMermaid } from "vitepress-plugin-mermaid";

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "Torch OpenCL",
  description: "A VitePress Site",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "Examples", link: "/markdown-examples" },
    ],

    sidebar: [
      {
        text: "Examples",
        items: [
          { text: "Markdown Examples", link: "/markdown-examples" },
          { text: "Runtime API Examples", link: "/api-examples" },
        ],
      },
      {
        text: "Architecture",
        items: [
          { text: "Overview", link: "/architecture/" },
          { text: "System Overview", link: "/architecture/system-overview" },
          { text: "Runtime Flow", link: "/architecture/runtime-flow" },
          { text: "Runtime Layer", link: "/architecture/runtime-layer" },
          { text: "Allocator Design", link: "/architecture/allocator-design" },
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
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
  },
});

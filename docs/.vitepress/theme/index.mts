import "./custom.css";
import { h, nextTick, watch } from "vue";
import type { Theme } from "vitepress";
import VPLTheme from "@lando/vitepress-theme-default-plus";
import { useData } from "vitepress";
import { createMermaidRenderer } from "vitepress-mermaid-renderer";

export default {
  extends: VPLTheme,
  Layout: () => {
    const { isDark } = useData();
    const initMermaid = () => {
      createMermaidRenderer({
        theme: isDark.value ? "dark" : "forest",
      });
    };
    nextTick(() => initMermaid());
    watch(
      () => isDark.value,
      () => {
        initMermaid();
      },
    );
    return h(VPLTheme.Layout);
  },
} satisfies Theme;

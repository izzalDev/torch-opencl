import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { parse } from "smol-toml";

// Load pyproject.toml
const pyprojectPath = path.resolve(__dirname, "../../pyproject.toml");
const pyprojectRaw = fs.readFileSync(pyprojectPath, "utf-8");
export const pyproject = parse(pyprojectRaw) as any;

/**
 * Follows PEP 440 and setuptools_scm guessing logic
 */
export const getVersion = () => {
  try {
    const raw = execSync("git describe --tags --always --long").toString().trim();
    const parts = raw.split("-");

    if (parts.length === 1 && !raw.startsWith("v")) {
      const commitCount = execSync("git rev-list --count HEAD").toString().trim();
      return `v0.1.0.dev${commitCount}`;
    }

    if (parts.length < 2) return raw.startsWith("v") ? raw : `v${raw}`;

    let tag = parts[0];
    const distance = parts[1];
    if (distance === "0") return tag.startsWith("v") ? tag : `v${tag}`;

    const preReleaseMatch = tag.match(/^(v?\d+\.\d+\.\d+)([abc]|rc)(\d+)$/);
    if (preReleaseMatch) {
      const [_, base, type, num] = preReleaseMatch;
      return `${base}${type}${parseInt(num) + 1}.dev${distance}`;
    }

    if (tag.includes(".dev")) {
      return `${tag.replace(/\.dev\d+$/, "")}.dev${distance}`;
    }

    const [major, minor, patch] = tag.replace("v", "").split(".");
    return `v${major}.${minor}.${parseInt(patch || "0") + 1}.dev${distance}`;
  } catch (e) {
    return "v0.1.0.dev0";
  }
};

/**
 * Injects pyproject.toml description into the home page tagline
 */
export const injectDynamicTagline = (pageData: any) => {
  if (pageData.frontmatter.layout === 'home') {
    pageData.frontmatter.hero = {
      ...pageData.frontmatter.hero,
      tagline: pyproject.project.description
    };
  }
};

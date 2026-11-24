/** @type {import('next').NextConfig} */
const nextConfig = {
  // Load pdf-parse's bundled pdf.js from node_modules at runtime instead of
  // bundling it (used by src/lib/pdf-loader.ts).
  serverExternalPackages: ["pdf-parse"],

  // 1. Whitelist external image domains here
  images: {
    domains: ["gravatar.com"],
  },

  async redirects() {
    return [
      {
        source: "/sign-in",
        destination: "/api/auth/login",
        permanent: true,
      },
      {
        source: "/sign-up",
        destination: "/api/auth/register",
        permanent: true,
      },
    ];
  },

  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
};

module.exports = nextConfig;

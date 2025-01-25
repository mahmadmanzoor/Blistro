import Navbar from "@/components/Navbar";
import Providers from "@/components/Providers";
import { cn, constructMetadata } from "@/lib/utils";
import { Inter } from "next/font/google";
import "./globals.css";

import "react-loading-skeleton/dist/skeleton.css";
import "simplebar-react/dist/simplebar.min.css";

import { Toaster } from "@/components/ui/toaster";

// 1. Load your font
const inter = Inter({ subsets: ["latin"] });

// 2. Export your metadata (make sure constructMetadata() doesn't add themeColor!)
export const metadata = {
  ...constructMetadata(),
  // e.g., { title: "Blistro", description: "A modern fullstack SaaS" }
};

// 3. Export a separate viewport object for themeColor
export const viewport = {
  themeColor: "#ffffff", // or any color you want
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="light">
      <Providers>
        <body
          className={cn(
            "min-h-screen font-sans antialiased grainy",
            inter.className
          )}
        >
          <Toaster />
          <Navbar />
          {children}
        </body>
      </Providers>
    </html>
  );
}

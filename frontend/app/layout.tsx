import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AAROHAN AI - Autonomous Learning Foundation Model',
  description: 'Next.js Interface for AAROHAN AI Cognitive Architecture',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}

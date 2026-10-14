# Social Media Content Creation System

This project generates daily social media content across LinkedIn, X, and Instagram in my brand voice, pulling topics from a content calendar in Air Table.

## How It Works

- The content calendar lives in Air Table
- Pull topics from the "Ready to Write" view
- Generated posts go into `/output/drafts/`
- Image assets are stored in `/output/assets/`

## Brand Voice

- Write like you're talking to one person over coffee, not presenting to a boardroom
- For full brand voice guide with examples, see `/docs/brandvoice.md`
- For platform-specific formatting rules, see `/docs/platform-rules.md`

## IMPORTANT Rules

- Never use m-dashes (—) — use regular dashes (-) instead
- Always replace the word "anti-gravity" with "AG"
- Never use technical jargon — explain everything in business terms
- Every post must teach something or share a real experience
- Be practical and actionable, not theoretical

## Mistakes to Avoid

- Air Table has both a "Status" and "Post Status" column — always use "Post Status"
- Do not publish or schedule anything directly — only create drafts
- When repurposing content, change the angle, not just shorten it

## How We Work

- Every file is named: `YYYY-MM-DD-platform-topic-slug.md`
- Never publish or schedule — drafts only
- When repurposing, change the angle, not just shorten
- Every time you write content, make sure you use the humanizer skill in `/skills/humanizer/`
- Check all output against banned words in `/docs/banned-words.txt`

## Skills

- `/skills/humanizer/` — Remove AI-generated writing patterns
- `/skills/brand-voice/` — Apply brand voice guidelines and examples

## Inker — IRB (Inker Rust Browser)

Tagline: Smudge the web, rethink the browser.

What is Inker / IRB?

Inker is a prototype concept for a new, Rust-based browser layer that reimagines how we interact with the web. IRB (Inker Browser) aims to:

Encourage intentional browsing and computation rather than mass-produced convenience.

Introduce the SMUDGE protocol — an experimental idea for encoding, protecting, and optionally monetizing content-processing operations.

Promote a Python-first app ecosystem inside the browser: browser-native apps built with/for Python 3.

This is an exploratory project — more manifesto than finished product. The goal is to provoke new ideas about privacy, infrastructure incentives, and how web clients should behave.

Key ideas

SMUDGE protocol — A proposed protocol that allows selective obfuscation/encoding of web content (a conceptual "smudge") so that certain computations or transformations are performed in ways that resist trivial packet interception or cloning. (Conceptual at this stage.)

Python-first apps — IRB will require Python 3 support so that browser apps are easy to write for data scientists and developers used to Python tooling.

Economics — The browser may include micro-payments (e.g., Bitcoin) to compensate third-party content processors (the “cleaners”) when performing compute or privacy-preserving transformations.

Goals

Build a minimal Rust browser shell that:

supports Python-embedded apps,

experiments with in-browser privacy primitives (SMUDGE),

explores micro-payment flows for compute/cleanup operations.

Provide a platform for experimenting with alternatives to the current “always-tracked”, convenience-first web.

Getting started (notes)

This repo currently contains:

A small Rust prototype exploring stateful behavior (see src/main.rs).

Conceptual notes on the SMUDGE protocol and architecture.

To run the Rust prototype:

Install Rust (stable).

Add rand = "0.8" to your Cargo.toml.

cargo run

Note: this repository is a work in progress and intentionally experimental.

License & contact

Author: Noel Garcia
If you’d like help turning this into a prototype, or if you want to collaborate on the SMUDGE ideas or Python integration, get in touch.

If you want, I can:

convert the README into a short website / landing page (HTML + CSS),

prepare a Cargo.toml and project layout for the code above,

add unit tests for core pieces (e.g., pipeclean and send/receive behaviors),

or refactor the code to be more modular (split into modules, add logging, CLI options to run N iterations instead of infinite).

Which of those would you like next? Or want me to just create a ready-to-run repo layout (Cargo.toml + src/main.rs + README.md) you can drop in and run?
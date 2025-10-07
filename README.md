## Inker — IRB (Inker Rust Browser)

A dialpadded web browser should be able to use scroll lock to use numpad for changing brick layout of browser math to increase speed and ability to code within the browser.

This browser aims to change the www. to irb. by introducing the SMUDGE protocol to a new browser on top of www. that, includes a bitcoin to pay your SMUDGE cleaning of the www. by using AI and the irb. protocol.

The SMUDGE protocol says information like on paper can SMUDGE your CPU to gear locks that can bypass brute force hacking through packet sniffing machine that generate SMUDGE to gear lock information in order to steal it over a networked cable or WIFI.

I plan to make IRB mostly to force browsers who use IRB to forcefully support Python 3. The browser I make will require Python to work and will have an ecosystem of Python first apps.

### Tagline: Smudge the web, rethink the browser.

What is Inker / IRB?

Inker is a prototype concept for a new, Rust-based browser layer that reimagines how we interact with the web. IRB (Inker Browser) aims to:

Encourage intentional browsing and computation rather than mass-produced convenience.

Introduce the SMUDGE protocol — an experimental idea for encoding, protecting, and optionally monetizing content-processing operations.

Promote a Python-first app ecosystem inside the browser: browser-native apps built with/for Python 3.

This is an exploratory project — more manifesto than finished product. The goal is to provoke new ideas about privacy, infrastructure incentives, and how web clients should behave.

### Key ideas

SMUDGE protocol — A proposed protocol that allows selective obfuscation/encoding of web content (a conceptual "smudge") so that certain computations or transformations are performed in ways that resist trivial packet interception or cloning. (Conceptual at this stage.)

Python-first apps — IRB will require Python 3 support so that browser apps are easy to write for data scientists and developers used to Python tooling.

Economics — The browser may include micro-payments (e.g., Bitcoin) to compensate third-party content processors (the “cleaners”) when performing compute or privacy-preserving transformations.

### Goals

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
# 🧠 Writing zkVM Programs with RISC Zero Bonsai: A Single-File Guide

## 📌 Overview

This document walks you through:
- Writing a RISC Zero **guest program**
- Writing a **host program** that interacts with Bonsai
- Building and uploading your method to Bonsai
- Generating and verifying proofs **in the cloud**

---

## ⚙️ 1. Prerequisites

Before anything else, install the following:

```bash
# RISC Zero CLI for scaffolding
cargo install cargo-risczero --version 0.20.0

# Bonsai CLI for uploading methods and managing jobs
cargo install bonsai-cli
```

Also, sign up at [https://bonsai.risczero.com](https://bonsai.risczero.com), get your **API key**, and save it locally:

```bash
bonsai config set api_key <your-api-key>
```

---

## 🛠️ 2. Create Your Project

Start fresh:

```bash
cargo risczero new hello_bonsai
cd hello_bonsai
```

You’ll now have:

```
hello_bonsai/
├── guest/         # Guest code (runs in zkVM)
├── host/          # Host code (calls Bonsai + verifies)
├── methods/       # Used to declare method ID
└── Cargo.toml
```

---

## 👾 3. Write the Guest Program

Edit `guest/src/main.rs`:

```rust
#![no_main]
#![no_std]

use risc0_zkvm::guest::env;

risc0_zkvm::guest::entry!(main);

pub fn main() {
    let input: u32 = env::read();
    let output = input * 2;
    env::commit(&output);
}
```

This zkVM guest:
- Reads an input `u32`
- Multiplies it by 2
- Commits the result to the journal (public output)

---

## 📦 4. Build the Guest ELF

```bash
cargo risczero build-elf
```

This creates the ELF binary at:  
```
guest/target/riscv-guest/release/guest
```

---

## ☁️ 5. Upload Method to Bonsai

Upload the compiled guest ELF:

```bash
bonsai upload \
  --name double_u32 \
  --path guest/target/riscv-guest/release/guest
```

After upload, Bonsai will return a **method ID**, e.g.:

```
Method ID: my-bonsai-user/double_u32@v1
```

Save this. You'll use it in the host.

---

## 🧑‍💻 6. Write the Host Program (Cloud Proving via Bonsai)

Edit `host/src/main.rs`:

```rust
use bonsai_sdk::alpha as bonsai;
use risc0_zkvm::{serde::to_vec, Receipt};
use std::env;

#[tokio::main]
async fn main() {
    // Replace this with your actual Bonsai method ID
    let method_id = "my-bonsai-user/double_u32@v1";

    // Input to the guest
    let input: u32 = 21;
    let input_data = to_vec(&input).unwrap();

    // Create the client from Bonsai environment variables
    let client = bonsai::Client::from_env().unwrap();

    // Upload input data
    let input_id = client.upload_input(input_data).await.unwrap();

    // Start the proof session
    let session = client
        .create_session()
        .elf(method_id)
        .input(input_id)
        .send()
        .await
        .unwrap();

    println!("Waiting for session: {}", session.uuid);

    // Wait until the session completes
    let receipt = client
        .session()
        .get_receipt(&session.uuid)
        .await
        .unwrap()
        .receipt;

    // Deserialize and print the output from the journal
    let output: u32 = risc0_zkvm::serde::from_slice(&receipt.journal).unwrap();
    println!("Output from zkVM: {}", output);

    // Verify proof locally
    receipt.verify(method_id).unwrap();
    println!("Proof verified ✅");
}
```

---

## 🧾 7. Add Dependencies

In `host/Cargo.toml`:

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
bonsai-sdk = { version = "0.3.1-alpha.0" }
risc0-zkvm = "0.20.0"
```

---

## ✅ 8. Run the Host Program

```bash
cargo run -p host
```

You’ll see:

```
Waiting for session: ...
Output from zkVM: 42
Proof verified ✅
```

You’ve now:
- Proved `21 * 2 = 42` inside zkVM
- Verified the proof
- Done it all with a remote prover (Bonsai)

---

## 🔐 9. How Bonsai Helps

| Feature              | Benefit |
|---------------------|---------|
| **Cloud proving**    | No local riscv emulation needed. Faster proofs. |
| **Hosted methods**   | Upload once, reuse across apps or frontend |
| **Job tracking**     | Async sessions allow scalable apps and batching |
| **Integration-ready**| Build APIs or smart contracts that verify proofs |

---

## 🔄 10. Optional Enhancements

- Use `env::read_vec()` for complex data
- Use `env::commit_slice()` to commit multiple values
- Sign your method to bind hash & ID using Bonsai CLI
- Use [Bonsai frontend SDKs](https://docs.risczero.com) (e.g., TypeScript) for web integration

---

## 📚 References

- [RISC Zero Dev Docs](https://dev.risczero.com/)
- [Bonsai CLI Docs](https://docs.bonsai.xyz/)
- [Bonsai SDK GitHub](https://github.com/risc0/bonsai-sdk)
- [Example zkApp with Bonsai](https://github.com/risc0/risc0/tree/main/examples/bonsai-starter)
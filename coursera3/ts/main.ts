// import { main } from "./kingCoin.ts";
// import { main } from "./kingCoin3.ts";
// import { main } from "./ticTacToe.ts";
// import { main } from "./FourInARow.ts";
import { main } from "./XInARow.ts";

// deno run  --allow-read --allow-write --v8-flags=--optimize-for-size,--max-old-space-size=196608,--max-semi-space-size=256,--semi-space-growth-factor=4 ./main.ts

// RES 5.5g
// MEM 8.5%

await main();

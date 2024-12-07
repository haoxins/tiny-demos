const wc = require("./witness_calculator.js");
const { readFileSync, writeFile } = require("node:fs");

const input = JSON.parse(readFileSync(process.argv[3], "utf8"));

const buffer = readFileSync(process.argv[2]);

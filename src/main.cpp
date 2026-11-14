// main.cpp - command line entry point of the CAC reference simulator.
#include <cstring>
#include <string>
#include <cstdio>

int run_all_tests();
int exp_matrix(const char* outdir);
int exp_invariance(const char* outdir);
int exp_ops(const char* outdir);
int exp_contention(const char* outdir);
int exp_sensitivity(const char* outdir);
int exp_train(const char* outdir);
int exp_threads(const char* outdir);
int exp_rare(const char* outdir);
int exp_dump(const char* outdir);

static void usage() {
  std::printf(
      "cac - Confluent Accumulation Cells reference simulator\n\n"
      "usage: cac <command> [output-directory]\n\n"
      "  test         run the registered test suite\n"
      "  matrix       standard 14-case matrix: reproduce the problem, evaluate all schemes\n"
      "  invariance   permutation / multi-device / combining invariance gates\n"
      "  ops          operation counts and state size per scheme\n"
      "  contention   compare-and-swap retries and warp-level combining under skew\n"
      "  sensitivity  CAC parameter grid (K, W, addends per destination, dynamic range)\n"
      "  train        downstream SGD divergence experiment\n"
      "  threads      real operating-system threads (not seeded, not reproducible by design)\n"
      "  rare         Monte Carlo: how often does an fp64 accumulator change the fp32 result\n"
      "  dump         write workloads and results for the independent Python cross-check\n"
      "  all          every experiment above except test\n\n"
      "The output directory defaults to ./results\n");
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 1; }
  const std::string cmd = argv[1];
  const char* out = (argc >= 3) ? argv[2] : "results";
  if (cmd == "test") return run_all_tests();
  if (cmd == "matrix") return exp_matrix(out);
  if (cmd == "invariance") return exp_invariance(out);
  if (cmd == "ops") return exp_ops(out);
  if (cmd == "contention") return exp_contention(out);
  if (cmd == "sensitivity") return exp_sensitivity(out);
  if (cmd == "train") return exp_train(out);
  if (cmd == "threads") return exp_threads(out);
  if (cmd == "rare") return exp_rare(out);
  if (cmd == "dump") return exp_dump(out);
  if (cmd == "all") {
    int rc = 0;
    rc |= exp_matrix(out);
    rc |= exp_invariance(out);
    rc |= exp_ops(out);
    rc |= exp_contention(out);
    rc |= exp_sensitivity(out);
    rc |= exp_train(out);
    rc |= exp_threads(out);
    rc |= exp_rare(out);
    return rc;
  }
  usage();
  return 1;
}

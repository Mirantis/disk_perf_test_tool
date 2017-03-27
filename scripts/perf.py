from wally import main
opts = "X -l DEBUG report /tmp/perf_tests/warm_doe".split()

def x():
    main.main(opts)

x()

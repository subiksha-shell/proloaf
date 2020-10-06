import sys

front_matter = "---\ndate: "\
                + str(sys.argv[1])\
                + "\ntitle: \"" + str(sys.argv[2])\
                + "\"\nlinkTitle: \"" + str(sys.argv[3])\
                + "\"\nresources:\n- src: \"./logs/**.{png,jpg}\"\n  title: \"Image #:counter\"\n  params:\n    byline:\"Results of train run\"\n---"
print(front_matter)

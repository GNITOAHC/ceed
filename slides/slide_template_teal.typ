// CP2 Slide Template - Forest Teal
// Same layout system as slide_template.typ; only the theme changes.

#import "/slide-kit/core.typ": *
#import "/slide-kit/themes.typ": forest-teal

#let theme = forest-teal
#let title-slide = title-slide.with(theme: theme)
#let contents-slide = contents-slide.with(theme: theme)
#let section-slide = section-slide.with(theme: theme)
#let content-slide = content-slide.with(theme: theme)
#let callout = callout.with(theme: theme)
#let card = card.with(theme: theme)
#let quiz-slide = quiz-slide.with(theme: theme)
#show: body => deck(theme: theme, body)

#title-slide(
  course: [Computer Programming II Lab],
  title: [Data Structures in Practice],
  details: [2026/07/22, Harry Wang],
  eyebrow: [FOREST TEAL TEMPLATE],
)

#contents-slide(items: (
  ([Motivation], [3]),
  ([Concept], [4]),
  ([Implementation], [6]),
  ([Practice], [8]),
))

#section-slide(
  title: [Concept],
  subtitle: [Build the mental model before showing the code.],
)

#content-slide(title: [A Clear Teaching Slide])[
  #two-columns(
    [
      == Core idea
      - Keep each slide focused.
      - Use examples students can trace.
      - Highlight one conclusion.
    ],
    [
      #callout(title: [Remember])[
        A good diagram or tiny code sample should remove one specific source of confusion.
      ]
    ],
    ratio: (1.15fr, 0.85fr),
  )
]

#content-slide(title: [Implementation])[
  #set text(size: 18pt)
  ```cpp
  #include <vector>

  int sum(const std::vector<int>& values) {
      int result = 0;
      for (int value : values) result += value;
      return result;
  }
  ```

  #v(0.35em)
  #callout(kind: "accent", title: [Common mistake])[
    State the failure mode directly, then show the corrected line.
  ]
]

#quiz-slide(
  number: 1,
  question: [Which statement best matches the lesson?],
  choices: (
    ([A], [A focused example is easier to debug.]),
    ([B], [Every slide needs five examples.]),
    ([C], [Long code is always more realistic.]),
    ([D], [Terminology should come before motivation.]),
  ),
  answer: [A],
)

import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.github.limmen",
      scalaVersion := "2.11.12",
      version      := "1.0.0"
    )),
    name := "example",
    libraryDependencies ++= Seq(
      scalaTest,
      mockito,
      sparkCore,
      sparkSql,
      sparkMlLib,
      sparkStreaming,
      scalaCsv,
      scallop,
      commonsMath,
      ftos
    ),
    test in assembly := {}
  )

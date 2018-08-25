import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.github.limmen",
      scalaVersion := "2.11.12",
      version      := "1.0.0"
    )),
    name := "fasttext_on_spark",
    libraryDependencies ++= Seq(
      scalaTest,
      mockito,
      sparkCore,
      sparkSql,
      sparkMlLib,
      sparkStreaming,
      scalaCsv,
      scallop,
      commonsMath
    ),
    test in assembly := {},
    crossPaths := false //remove version suffix from artifact
  )

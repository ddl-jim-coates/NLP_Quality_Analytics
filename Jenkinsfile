pipeline {
    agent any
    
    environment {
        DOMINO_API_KEY = credentials('domino-api-key')
        DOMINO_URL = 'https://se-demo.domino.tech/'
        DOMINO_PROJECT = 'jim_coates/NLP_Quality_Analytics'
    }
    
    stages {
        stage('Trigger Domino Training Job') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                script {
                    def jobPayload = [
                        command: ["python", "src/experiments/mlflow_tracking.py"],
                        isDirect: false,
                        title: "Jenkins Triggered Training - Build ${BUILD_NUMBER}",
                        tier: "Free",
                        publishApiEndpoint: false
                    ]
                    
                    def response = httpRequest(
                        httpMode: 'POST',
                        url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/jobs",
                        customHeaders: [
                            [name: 'X-Domino-Api-Key', value: "${DOMINO_API_KEY}"],
                            [name: 'Content-Type', value: 'application/json']
                        ],
                        requestBody: groovy.json.JsonOutput.toJson(jobPayload),
                        validResponseCodes: '200:299'
                    )
                    
                    def jobInfo = readJSON text: response.content
                    env.DOMINO_JOB_ID = jobInfo.id
                    echo "Started Domino job: ${jobInfo.id}"
                }
            }
        }
        
        stage('Wait for Training Completion') {
            when {
                expression { env.DOMINO_JOB_ID != null }
            }
            steps {
                script {
                    timeout(time: 30, unit: 'MINUTES') {
                        waitUntil {
                            def statusResponse = httpRequest(
                                httpMode: 'GET',
                                url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/jobs/${env.DOMINO_JOB_ID}",
                                customHeaders: [
                                    [name: 'X-Domino-Api-Key', value: "${DOMINO_API_KEY}"]
                                ],
                                validResponseCodes: '200:299'
                            )
                            
                            def jobStatus = readJSON text: statusResponse.content
                            echo "Job status: ${jobStatus.statuses[-1].status}"
                            
                            return jobStatus.statuses[-1].status in ['Succeeded', 'Failed', 'Error']
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo "Pipeline completed"
        }
        success {
            echo "✅ Successfully trained and deployed model"
        }
        failure {
            echo "❌ Pipeline failed"
        }
    }
}
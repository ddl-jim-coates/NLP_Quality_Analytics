pipeline {
    agent any
    
    environment {
        DOMINO_API_KEY = credentials('domino-api-key')
        DOMINO_URL = 'https://se-demo.domino.tech/'
        DOMINO_PROJECT = 'jim_coates/NLP_Quality_Analytics'
    }
    
    stages {
        stage('Debug Info') {
            steps {
                script {
                    echo "🔍 Debug Information:"
                    echo "Branch Name: ${env.BRANCH_NAME ?: 'NOT SET'}"
                    echo "Git Branch: ${env.GIT_BRANCH ?: 'NOT SET'}"
                    echo "Build Number: ${BUILD_NUMBER}"
                    echo "Domino URL: ${DOMINO_URL}"
                    echo "Domino Project: ${DOMINO_PROJECT}"
                }
            }
        }
        
        stage('Trigger Domino Training Job') {
            // REMOVED the 'when' condition so it always runs
            steps {
                script {
                    echo "🚀 Starting Domino job..."
                    
                    def jobPayload = [
                        command: ["python", "experiments/mlflow_tracking.py"],  // Fixed path - removed 'src/'
                        isDirect: false,
                        title: "Jenkins Triggered Training - Build ${BUILD_NUMBER}",
                        tier: "Free",
                        publishApiEndpoint: false
                    ]
                    
                    def response = httpRequest(
                        httpMode: 'POST',
                        url: "${DOMINO_URL}v1/projects/${DOMINO_PROJECT}/jobs",  // Fixed URL - removed extra slash
                        customHeaders: [
                            [name: 'X-Domino-Api-Key', value: "${DOMINO_API_KEY}"],
                            [name: 'Content-Type', value: 'application/json']
                        ],
                        requestBody: groovy.json.JsonOutput.toJson(jobPayload),
                        validResponseCodes: '200:299'
                    )
                    
                    def jobInfo = readJSON text: response.content
                    env.DOMINO_JOB_ID = jobInfo.id
                    echo "✅ Started Domino job: ${jobInfo.id}"
                }
            }
        }
        
        stage('Wait for Training Completion') {
            when {
                expression { env.DOMINO_JOB_ID != null }
            }
            steps {
                script {
                    echo "⏳ Waiting for job completion..."
                    timeout(time: 30, unit: 'MINUTES') {
                        waitUntil {
                            def statusResponse = httpRequest(
                                httpMode: 'GET',
                                url: "${DOMINO_URL}v1/projects/${DOMINO_PROJECT}/jobs/${env.DOMINO_JOB_ID}",
                                customHeaders: [
                                    [name: 'X-Domino-Api-Key', value: "${DOMINO_API_KEY}"]
                                ],
                                validResponseCodes: '200:299'
                            )
                            
                            def jobStatus = readJSON text: statusResponse.content
                            echo "📊 Job status: ${jobStatus.statuses[-1].status}"
                            
                            return jobStatus.statuses[-1].status in ['Succeeded', 'Failed', 'Error']
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo "🏁 Pipeline completed"
        }
        success {
            echo "✅ Successfully trained and deployed model"
        }
        failure {
            echo "❌ Pipeline failed"
        }
    }
}